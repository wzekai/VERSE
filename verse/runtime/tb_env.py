"""Interactive executor for Terminal-Bench tasks (the TB counterpart of executor.run_task).

SWE tasks are graded on a patch: the agent edits files and the tests run against its `git diff`.
Terminal-Bench (TB) tasks are graded on container state: the agent runs shell commands in the
task's own image, and the task is resolved if its tests pass on the final state. TB therefore has
its own session and scoring, but it uses the same tool-calling loop as executor.run_task (bash and
submit tools, region rotation, observation shrinking), so SWE and TB runs differ only in the task
family.

Container setup follows the official Terminal-Bench harness (DockerComposeManager):
  - TB1 tasks ship a docker-compose.yaml. The agent works in the `client` service (built from the
    task Dockerfile), and sibling services (databases, mock APIs, servers) provide the rest of the
    environment. The compose project is started with the same 8 T_BENCH_* variables the official
    harness sets; the agent's commands and the task's tests run in the client container. A plain
    `docker run` would drop the sibling services and compose settings such as dns/extra_hosts.
  - TB2.1 (Harbor) tasks have no compose file, only environment/Dockerfile, and use a plain build
    and run.
  - Terminal-Bench 4.0 (Harbor) tasks take sidecars from environment/docker-compose.yaml, with a
    runner-supplied `main` service. Their verifier is separate: tests/Dockerfile builds a verifier
    image that receives the task's declared `artifacts` from the agent and sidecar containers and
    runs /tests/test.sh (score() calls _score_separate).

Task definitions come from a terminal-bench checkout, one directory per task (e.g.
original-tasks/<id>/ with docker-compose.yaml, Dockerfile, run-tests.sh, tests/, solution.sh),
located via T2T_TB_TASKS.

Environment variables:
  T2T_TB_TASKS             task root (default /tmp/terminal-bench/original-tasks); a ':'-separated
                           list of roots, each optionally 'label=path' (tb1=...:tb2=...) to tell
                           apart tasks with the same name
  T2E_SWE_DOCKER           docker binary and prefix (default "docker"; use "sudo docker" only where
                           the runtime user is not in the docker group)
  T2E_SWE_DOCKER_RUN_ARGS  extra `docker run` flags (single-container path only, e.g. a sysctl)
  T2E_SWE_CONTAINER_CPUS   per-container CPU cap (default 6)
  T2E_TB_BUILD_TIMEOUT     per-task image build timeout in seconds (default 2400)
  T2E_SWE_EXEC_TIMEOUT     per-command exec timeout in seconds (default 120)
  T2E_TB_TEST_TIMEOUT      test-run timeout in seconds (default 1800)
  T2E_TB_ECR_REPO          optional shared registry repository for the cross-machine image cache
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid

logger = logging.getLogger(__name__)

_TB_WORKDIR = "/app"

# Seed system prompt for the blank TB executor harness (the TB counterpart of
# executor.BASE_PROMPT). TB tasks run in a terminal at /app with no git repo and no patch; a task
# is resolved when its own tests pass on the final container state. TB configurations select this
# prompt with `base_prompt: TB`. The SWE prompt's /testbed and "edit the source, do not just write
# a test" framing confuses executors on terminal tasks, which then end without running commands.
TB_BASE_PROMPT = (
    "You are an expert operating a Linux terminal. Your working directory is /app.\n"
    "You have a `bash` tool (runs shell commands; state persists across calls) and a `submit` tool.\n"
    "Workflow: (1) explore the environment (ls/cat/pwd) to understand what the task needs; "
    "(2) run the shell commands that accomplish the task — create/edit files, install packages, "
    "start services, produce outputs; (3) verify your work; (4) call `submit` only when the task is "
    "fully complete.\n"
    "Always call the bash tool to act — never just describe commands in prose. Do real work in the "
    "terminal before submitting."
)


def _tb_task_dir(instance_id: str, src: str = "") -> str:
    """Resolve a task id to its directory.

    T2T_TB_TASKS is a ':'-separated list of roots, each optionally labeled 'name=path'
    (e.g. 'tb1=/tmp/terminal-bench/original-tasks:tb2=/tmp/terminal-bench-2'). 88 task names
    exist in both TB1 and TB2.1 with different content, so instances carry `tb_src` to pick the
    labeled root. Without a matching label, the first root that contains the task is used."""
    spec = os.environ.get("T2T_TB_TASKS", "/tmp/terminal-bench/original-tasks")
    roots, keyed = [], {}
    for part in spec.split(":"):
        if "=" in part:
            k, v = part.split("=", 1)
            keyed[k] = v
            roots.append(v)
        else:
            roots.append(part)
    if src and src in keyed:
        return os.path.join(keyed[src], instance_id)
    for r in roots:
        d = os.path.join(r, instance_id)
        if os.path.isdir(d):
            return d
    return os.path.join(roots[0], instance_id)


_CONTENT_HASH_CACHE: dict = {}


def _task_content_hash(task_dir: str, max_file_bytes: int = 1_048_576) -> str:
    """Deterministic hash of a task directory, used as the image cache key.

    Same content gives the same image tag; any change to the Dockerfile, compose file or scripts
    gives a new tag and a rebuild. Hashes each file's relative path, size and first 1 MB of
    content (the size still catches changes past the cap). Files that do not enter the image
    (tests/, solution.sh) are included: a needless rebuild is cheap, while reusing a stale image
    would grade the task in the wrong environment. Cached per process, since a session is
    created per episode."""
    if task_dir in _CONTENT_HASH_CACHE:
        return _CONTENT_HASH_CACHE[task_dir]
    h = hashlib.sha1()
    if os.path.isdir(task_dir):
        for dirpath, dirs, files in sorted(os.walk(task_dir)):
            dirs.sort()
            for name in sorted(files):
                fp = os.path.join(dirpath, name)
                rel = os.path.relpath(fp, task_dir)
                try:
                    size = os.path.getsize(fp)
                    h.update(rel.encode())
                    h.update(str(size).encode())
                    with open(fp, "rb") as f:
                        h.update(f.read(max_file_bytes))
                except OSError:
                    h.update(rel.encode() + b"?")
    digest = h.hexdigest()[:10]
    _CONTENT_HASH_CACHE[task_dir] = digest
    return digest


class TBContainerSession:
    """A persistent Terminal-Bench task container.

    start() builds (or reuses) the task image and starts a long-lived container, execute(cmd)
    runs the agent's commands in it (state persists), score() runs the task's own tests in
    place, and close() tears it down. Like SWEContainerSession, methods return error strings or
    False instead of raising."""

    def __init__(self, instance_id: str, name: str | None = None, src: str = ""):
        self.instance_id = instance_id
        self.task_dir = _tb_task_dir(instance_id, src)
        self.name = name or f"t2e_tb_{uuid.uuid4().hex[:12]}"
        # Docker tags and compose project names must match [a-z0-9_.-], so sanitize the id.
        # The image tag is stable per task and content-addressed (hash of the task files): the
        # image is built once, reused by later episodes, and can be shared across machines
        # through a registry. If the task files change, the hash changes and the stale image is
        # never reused; a path-based key would collide across checkouts at the same path. A
        # per-session tag would rebuild the image every episode, which dominates TB wall-clock
        # time. Containers and compose projects stay unique per episode.
        task_slug = re.sub(r"[^a-z0-9]+", "_", instance_id.lower()).strip("_")[:60] or "tb"
        sess_slug = f"{task_slug}_{uuid.uuid4().hex[:8]}"
        self.img = f"tb_img_{task_slug}_{_task_content_hash(self.task_dir)}"
        self.project = f"tb_{sess_slug}"              # compose project (per-episode isolation)
        self.client = f"{self.project}_client"        # client container name
        self.docker = os.environ.get("T2E_SWE_DOCKER", "docker")
        self.exec_timeout = int(os.environ.get("T2E_SWE_EXEC_TIMEOUT", "120"))
        self.workdir = _TB_WORKDIR          # set in start() from the container's WORKDIR
        self.compose_path = os.path.join(self.task_dir, "docker-compose.yaml")
        # Harbor layout (Terminal-Bench 4.0): sidecar services live in
        # environment/docker-compose.yaml, and the runner supplies the agent's `main` service
        # (Harbor's docker-compose-build.yaml); _compose() prepends that overlay. TB1 tasks keep
        # their root-level docker-compose.yaml.
        self.harbor_compose = (not os.path.exists(self.compose_path) and os.path.exists(
            os.path.join(self.task_dir, "environment", "docker-compose.yaml")))
        if self.harbor_compose:
            self.compose_path = os.path.join(self.task_dir, "environment", "docker-compose.yaml")
        self.use_compose = os.path.exists(self.compose_path)
        self.exec_target = self.client if self.use_compose else self.name
        self.last_test_output = ""          # test output of the last score() (failure signature)
        self._started = False

    def _run(self, args: str, timeout: int) -> subprocess.CompletedProcess:
        """Run a `<docker> <args>` command (prepends the docker binary)."""
        return self._run_raw(f"{self.docker} {args}", timeout)

    def _run_raw(self, cmd: str, timeout: int) -> subprocess.CompletedProcess:
        """Run a full command line in its own process group, so a timed-out `sudo docker` child
        can be killed (as in SWEContainerSession._run: an unprivileged wait() on a root child
        otherwise blocks forever). Takes a full command line because compose commands carry an
        environment-variable prefix."""
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout)
            return subprocess.CompletedProcess(cmd, p.returncode, out, err)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(p.pid)
                subprocess.run(f"sudo -n kill -9 -{pgid} 2>/dev/null || kill -9 -{pgid}",
                               shell=True, capture_output=True, timeout=30)
            except Exception:
                pass
            try:
                p.communicate(timeout=10)
            except Exception:
                pass
            raise

    def _build_context(self) -> str | None:
        """Directory with the task's Dockerfile for the single-container path (no compose): the
        task root (some TB1 tasks) or environment/ (TB2/Harbor). None if the task cannot be built
        (the gold scan excludes such tasks)."""
        if os.path.exists(os.path.join(self.task_dir, "Dockerfile")):
            return self.task_dir
        env_dir = os.path.join(self.task_dir, "environment")
        if os.path.exists(os.path.join(env_dir, "Dockerfile")):
            return env_dir
        return None

    def _task_toml(self) -> dict:
        """Parsed task.toml (Harbor task manifest); {} when absent or unparseable."""
        p = os.path.join(self.task_dir, "task.toml")
        if not os.path.exists(p):
            return {}
        try:
            import tomllib
            return tomllib.load(open(p, "rb"))
        except Exception as e:
            logger.warning(f"[tb_env] task.toml parse failed ({self.instance_id}): {e}")
            return {}

    def _prebuilt_image(self) -> str | None:
        """task.toml [environment].docker_image: a registry reference to the task's prebuilt
        image. DeepSWE ships one per task; its Dockerfile clones from GitHub at build time, so
        pulling the published image is faster and avoids that clone."""
        return (self._task_toml().get("environment", {}) or {}).get("docker_image") or None

    def _agent_network_none(self) -> bool:
        """True when task.toml [agent].network_mode == 'no-network': the agent container then
        runs with --network none. DeepSWE needs this: the repository's later history is removed
        from the image, but the real fix still exists upstream, and network access would let
        the agent fetch it."""
        return (self._task_toml().get("agent", {}) or {}).get("network_mode") == "no-network"

    def _verifier_reset_cmd(self) -> str | None:
        """Command that resets the repo to its base commit before in-place verification.

        Applies when task.toml sets [verifier].environment_mode == 'separate' and a base commit
        (DeepSWE). The official verifier applies the collected patch in a fresh base-state
        container. Here the tests run in the agent's container, so after collection (the patch
        is already captured) the repo is hard-reset to the base commit. Otherwise a patch that
        creates a file fails `git apply` ("already exists in working directory").
        `git clean -fd` (no -x) removes the agent's untracked files but keeps ignored build
        state (node_modules, ...). None when the task does not need a reset."""
        cfg = self._task_toml()
        if (cfg.get("verifier", {}) or {}).get("environment_mode") != "separate":
            return None
        base = (cfg.get("metadata", {}) or {}).get("base_commit_hash")
        if not base:
            return None
        return (f"cd {self.workdir} && git config --global --add safe.directory "
                f"{self.workdir} && git reset --hard {base} && git clean -fd")

    def _compose_env(self) -> dict:
        """Variables for compose-file interpolation.

        TB1: the 8 T_BENCH_* variables the official DockerComposeManager sets (fixed paths plus
        this session's client image and container names). The host-side log dirs are created
        here because the compose file bind-mounts them. Harbor: its own variables (below)."""
        if self.harbor_compose:
            # Harbor's compose variables (docker.py ComposeInfraEnvVars): image and build context of
            # the `main` service; CPUS applies the same per-container cap as the single-container
            # path.
            return {"MAIN_IMAGE_NAME": f"{self.img}_client", "MAIN_CONTAINER_NAME": self.client,
                    "CONTEXT_DIR": os.path.dirname(self.compose_path),
                    "CPUS": os.environ.get("T2E_SWE_CONTAINER_CPUS", "6")}
        logs = "/tmp/" + self.project + "_logs"
        alogs = "/tmp/" + self.project + "_agent_logs"
        os.makedirs(logs, exist_ok=True)
        os.makedirs(alogs, exist_ok=True)
        return {
            # image name keyed on the task (self.img), not the episode, so compose build is a
            # cache hit after the first episode of each task on a machine
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"{self.img}_client",
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": self.client,
            "T_BENCH_TASK_DOCKER_NAME_PREFIX": self.project,
            "T_BENCH_TEST_DIR": "/tests",
            "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
            "T_BENCH_TASK_LOGS_PATH": logs,
            "T_BENCH_TASK_AGENT_LOGS_PATH": alogs,
        }

    def _compose(self, sub: str) -> str:
        """Compose command string for subcommand `sub`.

        Exports the compose variables for file interpolation and, when docker runs under sudo,
        forwards them with `sudo -E` (`docker compose` reads them from its own process env, which
        sudo otherwise strips). Same form as the official harness:
        `docker compose -p <proj> -f <file> <sub>`."""
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in self._compose_env().items())
        docker = self.docker
        if docker.strip().startswith("sudo") and " -E" not in docker:
            docker = docker.replace("sudo", "sudo -E", 1)
        files = f"-f {shlex.quote(self.compose_path)}"
        # Harbor: the project directory is environment/ (sidecar build contexts are relative to
        # it); the `main` overlay comes first and the task's compose file last.
        if self.harbor_compose:
            files = (f"--project-directory {shlex.quote(os.path.dirname(self.compose_path))} "
                     f"-f {shlex.quote(self._harbor_overlay())} {files}")
        return f"{exports} {docker} compose -p {shlex.quote(self.project)} {files} {sub}"

    # Harbor's src/harbor/environments/docker/docker-compose-build.yaml, plus the image and
    # container names and the CPU cap this runner controls (Harbor sets those via its overrides).
    _HARBOR_MAIN_OVERLAY = (
        "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n    container_name: ${MAIN_CONTAINER_NAME}\n"
        "    build:\n      context: ${CONTEXT_DIR}\n    pull_policy: build\n"
        "    command: [\"sh\", \"-c\", \"sleep infinity\"]\n    cpus: ${CPUS}\n")

    def _harbor_overlay(self) -> str:
        """Per-session file holding the `main` service definition Harbor merges under the task's
        environment/docker-compose.yaml. Recreated on demand; removed by close()."""
        p = os.path.join(tempfile.gettempdir(), f"{self.project}_main.yaml")
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(self._HARBOR_MAIN_OVERLAY)
        return p

    def _resolve_workdir(self, container: str) -> None:
        insp = self._run(f"inspect -f {shlex.quote('{{.Config.WorkingDir}}')} {shlex.quote(container)}",
                         timeout=60)
        wd = (insp.stdout or "").strip()
        self.workdir = wd if wd else _TB_WORKDIR

    def start(self) -> bool:
        """Bring the task environment up.

        Compose tasks: build the images and run `compose up`; the agent works in the client
        service. Single-Dockerfile tasks (Harbor): build or pull one image and run one sleeping
        container. The container's own WORKDIR is used (some tasks live under /app/<repo> or
        /home/<user>). Images are cached per task, as in Harbor, because building them dominates
        TB wall-clock time."""
        if not os.path.isdir(self.task_dir):
            logger.warning(f"[tb_env] no task dir for {self.instance_id} at {self.task_dir}")
            return False
        build_to = int(os.environ.get("T2E_TB_BUILD_TIMEOUT", "2400"))
        try:
            if self.use_compose:
                # image cache: local image, else registry pull, else build. `build` also needs
                # the compose variables because image names are templated.
                ok = self._pull_or_build(
                    f"{self.img}_client",
                    lambda: self._run_raw(self._compose("build"),
                                          timeout=build_to).returncode == 0)
                if not ok:
                    logger.warning(f"[tb_env] compose image unavailable ({self.instance_id})")
                    return False
                # Harbor starts its stacks with `up --wait` (healthchecks, depends_on conditions)
                u = (self._run_raw(self._compose("up -d --wait"), timeout=build_to) if self.harbor_compose
                     else self._run_raw(self._compose("up -d"), timeout=600))
                if u.returncode != 0:
                    logger.warning(f"[tb_env] compose up failed ({self.instance_id}): {u.stderr[-300:]}")
                    self._compose_down()
                    return False
                # confirm the client container is up before declaring success
                chk = self._run(f"inspect -f {shlex.quote('{{.State.Running}}')} {shlex.quote(self.client)}",
                                timeout=60)
                if "true" not in (chk.stdout or "").lower():
                    logger.warning(f"[tb_env] client container not running ({self.instance_id})")
                    self._compose_down()
                    return False
                self._resolve_workdir(self.client)
                self._started = True
                return True
            # Single-container (Harbor) path. Image source: the task's published image when
            # task.toml names one (pull), else the local Dockerfile (build), both behind the
            # same cache.
            prebuilt = self._prebuilt_image()
            if prebuilt:
                make_image = lambda: (
                    self._run(f"pull -q {shlex.quote(prebuilt)}", timeout=build_to).returncode == 0
                    and self._run(f"tag {shlex.quote(prebuilt)} {shlex.quote(self.img)}",
                                  timeout=60).returncode == 0)
            else:
                ctx = self._build_context()
                if ctx is None:
                    logger.warning(f"[tb_env] no Dockerfile/compose for {self.instance_id}")
                    return False
                # Stable per-task tag: later episodes (and, with a shared registry, other
                # machines) reuse the image. Concurrent builds of the same task are safe: both
                # produce the same image under the same tag.
                make_image = lambda: self._run(
                    f"build -t {shlex.quote(self.img)} {shlex.quote(ctx)}",
                    timeout=build_to).returncode == 0
            ok = self._pull_or_build(self.img, make_image)
            if not ok:
                logger.warning(f"[tb_env] image unavailable ({self.instance_id})")
                return False
            extra = os.environ.get("T2E_SWE_DOCKER_RUN_ARGS", "").strip()
            if self._agent_network_none():
                extra = (extra + " --network none").strip()
            cpus = os.environ.get("T2E_SWE_CONTAINER_CPUS", "6")
            insp = self._run(f"image inspect -f {shlex.quote('{{.Config.WorkingDir}}')} "
                             f"{shlex.quote(self.img)}", timeout=60)
            wd = (insp.stdout or "").strip()
            self.workdir = wd if wd else _TB_WORKDIR
            cp = self._run(
                f"run -d --name {shlex.quote(self.name)} {extra} --cpus={cpus} "
                f"-w {shlex.quote(self.workdir)} {shlex.quote(self.img)} sleep 86400",
                timeout=300,
            )
            if cp.returncode != 0:
                logger.warning(f"[tb_env] container start failed ({self.instance_id}): {cp.stderr[-300:]}")
                try:
                    self._run(f"rm -f {shlex.quote(self.name)}", timeout=60)
                except Exception:
                    pass
                return False
            self._started = True
            return True
        except Exception as e:
            logger.warning(f"[tb_env] start exception ({self.instance_id}): {e}")
            if self.use_compose:
                self._compose_down()
            return False

    def _ecr_ref(self, local_img: str) -> str | None:
        """Cross-machine cache reference for a local image tag: <T2E_TB_ECR_REPO>:<local_tag>.
        None when the variable is unset (only the machine-local cache applies)."""
        repo = os.environ.get("T2E_TB_ECR_REPO", "").strip()
        return f"{repo}:{local_img}" if repo else None

    def _pull_or_build(self, local_img: str, build_fn) -> bool:
        """Two-level image cache: use the local image if present, else pull it from the shared
        registry, else build it locally and push it (best effort)."""
        if self._run(f"image inspect {shlex.quote(local_img)}", timeout=60).returncode == 0:
            return True
        ref = self._ecr_ref(local_img)
        if ref:
            p = self._run(f"pull -q {shlex.quote(ref)}", timeout=1800)
            if p.returncode == 0:
                self._run(f"tag {shlex.quote(ref)} {shlex.quote(local_img)}", timeout=60)
                return True
        if not build_fn():
            return False
        if ref:            # best-effort push: a race or permission failure must not fail the task
            self._run(f"tag {shlex.quote(local_img)} {shlex.quote(ref)}", timeout=60)
            self._run(f"push -q {shlex.quote(ref)}", timeout=1800)
        return True

    def prebuild(self) -> bool:
        """Build or fetch this task's images without starting containers.

        Uses the same cache as start() (local image, registry pull, then build and push) and
        also prepares the separate verifier image when the task has one. Run at machine startup
        so sweeps do not wait on builds; optional, since start() resolves images lazily."""
        if not os.path.isdir(self.task_dir):
            return False
        build_to = int(os.environ.get("T2E_TB_BUILD_TIMEOUT", "2400"))
        try:
            if self._separate_verifier() and not self._verifier_image(build_to):
                return False
            if self.use_compose:
                return self._pull_or_build(
                    f"{self.img}_client",
                    lambda: self._run_raw(self._compose("build"),
                                          timeout=build_to).returncode == 0)
            prebuilt = self._prebuilt_image()
            if prebuilt:
                return self._pull_or_build(
                    self.img,
                    lambda: (self._run(f"pull -q {shlex.quote(prebuilt)}",
                                       timeout=build_to).returncode == 0
                             and self._run(f"tag {shlex.quote(prebuilt)} {shlex.quote(self.img)}",
                                           timeout=60).returncode == 0))
            ctx = self._build_context()
            if ctx is None:
                return False
            return self._pull_or_build(
                self.img,
                lambda: self._run(f"build -t {shlex.quote(self.img)} {shlex.quote(ctx)}",
                                  timeout=build_to).returncode == 0)
        except Exception as e:
            logger.warning(f"[tb_env] prebuild failed ({self.instance_id}): {e}")
            return False

    def execute(self, command: str) -> str:
        """Run one shell command in the task container's working directory. Returns combined
        stdout and stderr, truncated like the SWE session so the agent gets the same observation
        budget."""
        from verse.runtime.swe_env import truncate_observation
        if not self._started:
            return "ERROR: container not started"
        wrapped = f"cd {shlex.quote(self.workdir)} && {command}"
        try:
            cp = self._run(f"exec {shlex.quote(self.exec_target)} bash -lc {shlex.quote(wrapped)}",
                           timeout=self.exec_timeout)
            out = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
            return truncate_observation(out)
        except subprocess.TimeoutExpired:
            return (f"ERROR: command timed out after {self.exec_timeout}s. Do not re-run "
                    "the same command — narrow its scope or background long-running work "
                    "so each command fits the limit.")
        except Exception as e:
            return f"ERROR: {e}"

    def _verifier_collect_cmds(self) -> list[tuple[str, int]]:
        """[[verifier.collect]] commands from task.toml (Harbor): artifact-collection steps that
        run in the agent's container before the tests (e.g. DeepSWE's
        `git diff base HEAD > /logs/artifacts/model.patch`, so only committed work is graded).
        [] for tasks without a task.toml or a collect section (TB1, TB2, tblite)."""
        p = os.path.join(self.task_dir, "task.toml")
        if not os.path.exists(p):
            return []
        try:
            import tomllib
            cfg = tomllib.load(open(p, "rb"))
        except Exception as e:
            logger.warning(f"[tb_env] task.toml parse failed ({self.instance_id}): {e}")
            return []
        out = []
        for step in (cfg.get("verifier", {}) or {}).get("collect", []) or []:
            cmd = step.get("command")
            if cmd:
                out.append((cmd, int(step.get("timeout_sec", 300))))
        return out

    def _phi_from_reward_json(self, ctr: str | None = None) -> float | None:
        """Verdict from /logs/verifier/reward.json ({"reward": 0|1}), Harbor's allowlist grade
        (DeepSWE: only the task's fail-to-pass and pass-to-pass node ids count). It takes
        precedence over the pytest summary, because a DeepSWE Python suite prints a whole-suite
        summary that can disagree with the allowlist grade. Absent on TB1, TB2, tblite and
        tb-pro."""
        r = self._run(f"exec {shlex.quote(ctr or self.exec_target)} cat /logs/verifier/reward.json",
                      timeout=60)
        if r.returncode == 0 and (r.stdout or "").strip():
            try:
                return 1.0 if int(json.loads(r.stdout)["reward"]) == 1 else 0.0
            except Exception:
                return None
        return None

    def _phi_from_reward_txt(self, ctr: str | None = None) -> float | None:
        """Verdict from /logs/verifier/reward.txt (0|1|-1), the coarse runner verdict. tb-pro
        writes it from the pytest exit code; -1 marks a runner crash and counts as an infra
        failure. Fallback for runners whose output has no parseable pytest summary."""
        r = self._run(f"exec {shlex.quote(ctr or self.exec_target)} cat /logs/verifier/reward.txt",
                      timeout=60)
        if r.returncode == 0 and (r.stdout or "").strip():
            v = r.stdout.strip().split()[0]
            try:
                fv = float(v)
            except ValueError:
                return None
            if fv >= 1.0:
                return 1.0
            if fv >= 0.0:
                return 0.0
        return None                              # -1, unparseable or absent: no verdict

    def score(self, run_id: str | None = None) -> float | None:
        """Run the task's own tests on the current container state and return the verdict.

        Returns 1.0 (resolved), 0.0 (not resolved) or None (infra failure). Verdict precedence:
        reward.json (allowlist-graded, DeepSWE), then the pytest -rA summary (SWE-bench rule: at
        least one test ran and none failed or errored; used for runners that do not write
        reward.json), then reward.txt (coarse runner exit, tb-pro's non-pytest suites). The tests
        are copied in only now, never during the agent's turns, so the agent cannot read or
        modify them."""
        if not self._started:
            return None
        if self._separate_verifier():
            return self._score_separate()
        # runner: <task>/run-tests.sh (TB1) or <task>/tests/test.sh (TB2/Harbor)
        runner = os.path.join(self.task_dir, "run-tests.sh")
        if not os.path.exists(runner):
            runner = os.path.join(self.task_dir, "tests", "test.sh")
            if not os.path.exists(runner):
                logger.warning(f"[tb_env] no test runner for {self.instance_id}")
                return None
        tgt = self.exec_target
        try:
            self._run(f"exec {shlex.quote(tgt)} mkdir -p /tests /logs/verifier /logs/artifacts",
                      timeout=60)
            for cmd, cto in self._verifier_collect_cmds():
                self._run(f"exec {shlex.quote(tgt)} bash -lc {shlex.quote(cmd)}", timeout=cto)
            reset = self._verifier_reset_cmd()   # separate-verifier tasks: reset to base state
            if reset:
                self._run(f"exec {shlex.quote(tgt)} bash -lc {shlex.quote(reset)}", timeout=300)
            # copy the tests and runner in only now, after the agent's turns
            self._run(f"cp {shlex.quote(os.path.join(self.task_dir, 'tests'))}/. "
                      f"{shlex.quote(tgt)}:/tests/", timeout=120)
            self._run(f"cp {shlex.quote(runner)} {shlex.quote(tgt)}:/run-tests.sh", timeout=120)
            t = self._run(f"exec -e TEST_DIR=/tests {shlex.quote(tgt)} bash /run-tests.sh",
                          timeout=int(os.environ.get("T2E_TB_TEST_TIMEOUT", "1800")))
            out = (t.stdout or "") + "\n" + (t.stderr or "")
            self.last_test_output = out          # kept as the failure signature for attribution
            phi = self._phi_from_reward_json()
            if phi is None:
                phi = _phi_from_pytest(out)
            return phi if phi is not None else self._phi_from_reward_txt()
        except subprocess.TimeoutExpired:
            logger.warning(f"[tb_env] test run timed out ({self.instance_id})")
            return None
        except Exception as e:
            logger.warning(f"[tb_env] score failed ({self.instance_id}): {e}")
            return None

    # Harbor separate verifier (Terminal-Bench 4.0)
    def _separate_verifier(self) -> bool:
        """True when task.toml sets [verifier].environment_mode == 'separate' and tests/Dockerfile
        exists: the tests run in their own image, and the task's declared `artifacts` are copied
        over from the agent environment (Harbor's trial flow). DeepSWE also declares separate
        mode but has a base commit, so it uses the in-place reset (_verifier_reset_cmd)."""
        cfg = self._task_toml()
        return ((cfg.get("verifier", {}) or {}).get("environment_mode") == "separate"
                and os.path.exists(os.path.join(self.task_dir, "tests", "Dockerfile"))
                and self._verifier_reset_cmd() is None)

    def _verifier_image(self, build_to: int) -> str | None:
        """Verifier image built from <task>/tests/Dockerfile, with the same cache and
        content-addressed tag family as the task image. None if unavailable."""
        img = f"{self.img}_verifier"
        ctx = os.path.join(self.task_dir, "tests")
        ok = self._pull_or_build(img, lambda: self._run(
            f"build -t {shlex.quote(img)} {shlex.quote(ctx)}", timeout=build_to).returncode == 0)
        return img if ok else None

    def _svc(self, service: str | None) -> str:
        """Container for a compose service name (Harbor: None or 'main' is the agent's
        container); '' when a named sidecar cannot be resolved."""
        if not service or service == "main":
            return self.exec_target
        if not self.harbor_compose:
            return ""
        r = self._run_raw(self._compose(f"ps -q {shlex.quote(service)}"), timeout=60)
        ids = (r.stdout or "").strip().splitlines()
        return ids[0].strip() if ids else ""

    def _score_separate(self) -> float | None:
        """Harbor 'separate' verifier, following harbor/trial/single_step.py.

        [[verifier.collect]] hooks run in their service. The declared artifacts (plus the
        implicit /logs/artifacts) are copied out of the agent container first; main is then
        stopped before collecting from sidecars. The agent environment is shut down, and a
        verifier container built from tests/Dockerfile receives the artifacts at their original
        paths and runs /tests/test.sh with [verifier.env]. Verdict: /logs/verifier/reward.json,
        else reward.txt (Harbor's precedence)."""
        import shutil
        cfg = self._task_toml()
        ver = cfg.get("verifier", {}) or {}
        venv = ver.get("environment") or cfg.get("environment", {}) or {}
        arts = [dict(a) if isinstance(a, dict) else {"source": a} for a in (cfg.get("artifacts") or [])]
        if not any(a.get("source", "").rstrip("/") == "/logs/artifacts" and a.get("service") in (None, "main")
                   for a in arts):
            arts.insert(0, {"source": "/logs/artifacts"})     # Harbor's implicit convention entry
        hooks = [h for h in (ver.get("collect", []) or []) if h.get("command")]
        sidecars = ({a.get("service") for a in arts} | {h.get("service") for h in hooks}) - {None, "main"}
        host = tempfile.mkdtemp(prefix=f"{self.project}_artifacts_")
        vname = f"{self.project}_verifier"

        def _collect(services: set) -> None:
            for h in hooks:                                     # collect hooks, best-effort
                svc = h.get("service") or "main"
                ctr = self._svc(svc) if svc in services else ""
                if not ctr:
                    continue
                sh = "bash -c" if svc == "main" else "sh -c"    # sidecars may lack bash (Harbor)
                self._run(f"exec {shlex.quote(ctr)} {sh} {shlex.quote(h['command'])}",
                          timeout=int(float(h.get("timeout_sec", 300))))
            for a in arts:                                      # artifacts, best-effort
                svc = a.get("service") or "main"
                ctr = self._svc(svc) if svc in services else ""
                src = a["source"].rstrip("/") or "/"
                dst = os.path.join(host, src.lstrip("/"))
                if not ctr or os.path.exists(dst):              # no container / overlap
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if a.get("exclude"):
                    ex = " ".join(f"--exclude={shlex.quote(x)}" for x in a["exclude"])
                    os.makedirs(dst, exist_ok=True)
                    self._run_raw(f"{self.docker} exec {shlex.quote(ctr)} tar -C {shlex.quote(src)} "
                                  f"-cf - {ex} . | tar -xf - -C {shlex.quote(dst)}", timeout=900)
                else:
                    self._run(f"cp {shlex.quote(ctr)}:{shlex.quote(src)} {shlex.quote(dst)}", timeout=900)

        try:
            build_to = int(os.environ.get("T2E_TB_BUILD_TIMEOUT", "2400"))
            self._run(f"exec {shlex.quote(self.exec_target)} mkdir -p /logs/verifier /logs/artifacts",
                      timeout=60)
            _collect({"main"})
            if sidecars:
                self._run_raw(self._compose("stop main"), timeout=180)   # Harbor: main stops first
                _collect(sidecars)
            self.close()                          # separate mode: the agent environment is done
            vimg = self._verifier_image(build_to)
            if not vimg:
                logger.warning(f"[tb_env] verifier image unavailable ({self.instance_id})")
                return None
            # Verifier resources and network follow task.toml ([verifier.environment], else
            # [environment]). The network is Harbor's default (public), or none when disallowed.
            net = (" --network none" if venv.get("allow_internet") is False
                   or venv.get("network_mode") == "no-network" else "")
            cpus = venv.get("cpus") or os.environ.get("T2E_SWE_CONTAINER_CPUS", "6")
            cp = self._run(f"run -d --name {shlex.quote(vname)}{net} --cpus={cpus} "
                           f"{shlex.quote(vimg)} sleep 86400", timeout=300)
            if cp.returncode != 0:
                logger.warning(f"[tb_env] verifier start failed ({self.instance_id}): {cp.stderr[-300:]}")
                return None
            # Harbor's ensure_dirs/empty_dirs always `chmod 777` the dirs they create; verifier
            # scripts that drop privileges (su postgres, runuser nobody, sage-preparse temp files)
            # rely on that.
            self._run(f"exec {shlex.quote(vname)} sh -c {shlex.quote('mkdir -p /logs/verifier /logs/artifacts && chmod 777 /logs/verifier /logs/artifacts')}",
                      timeout=60)
            for a in arts:                        # restore at the original paths
                src = a["source"].rstrip("/") or "/"
                hp = os.path.join(host, src.lstrip("/"))
                if not os.path.exists(hp):
                    continue
                if os.path.isdir(hp):             # Harbor empty_dirs + upload_dir
                    self._run(f"exec {shlex.quote(vname)} sh -c {shlex.quote(f'mkdir -p {shlex.quote(src)} && find {shlex.quote(src)} -mindepth 1 -delete && chmod 777 {shlex.quote(src)}')}",
                              timeout=120)
                    self._run(f"cp {shlex.quote(hp)}/. {shlex.quote(vname)}:{shlex.quote(src)}", timeout=900)
                else:                             # Harbor ensure_dirs(parent) + upload_file
                    parent = shlex.quote(os.path.dirname(src))
                    self._run(f"exec {shlex.quote(vname)} sh -c {shlex.quote(f'mkdir -p {parent} && chmod 777 {parent}')}",
                              timeout=60)
                    self._run(f"cp {shlex.quote(hp)} {shlex.quote(vname)}:{shlex.quote(src)}", timeout=900)
            env = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in (ver.get("env") or {}).items())
            to = int(float(ver.get("timeout_sec") or os.environ.get("T2E_TB_TEST_TIMEOUT", "1800")))
            t = self._run(f"exec {env} {shlex.quote(vname)} bash -c 'chmod +x /tests/test.sh; /tests/test.sh'",
                          timeout=to)
            self.last_test_output = (t.stdout or "") + "\n" + (t.stderr or "")
            phi = self._phi_from_reward_json(vname)
            return phi if phi is not None else self._phi_from_reward_txt(vname)
        except subprocess.TimeoutExpired:
            logger.warning(f"[tb_env] separate verifier timed out ({self.instance_id})")
            return None
        except Exception as e:
            logger.warning(f"[tb_env] separate verifier failed ({self.instance_id}): {e}")
            return None
        finally:
            try:
                self._run(f"rm -f {shlex.quote(vname)}", timeout=120)
            except Exception:
                pass
            shutil.rmtree(host, ignore_errors=True)

    def _compose_down(self) -> None:
        try:
            # remove containers, networks and volumes; keep the images for later episodes
            self._run_raw(self._compose("down -v"), timeout=300)
        except Exception:
            pass

    def close(self) -> None:
        if not self._started and not self.use_compose:
            return
        try:
            if self.use_compose:
                self._compose_down()
            else:
                # remove the container; the per-task image stays for later episodes (disk use is
                # bounded by the number of tasks)
                self._run(f"rm -f {shlex.quote(self.name)}", timeout=120)
        except Exception:
            pass
        if self.harbor_compose:
            try:
                os.remove(os.path.join(tempfile.gettempdir(), f"{self.project}_main.yaml"))
            except OSError:
                pass
        self._started = False


def _phi_from_pytest(output: str) -> float | None:
    """Verdict from a pytest -rA summary: resolved iff at least one test ran and none failed or
    errored (SWE-bench rule). None if there is no parseable result (an infra failure, not a model
    verdict)."""
    m_pass = re.search(r"(\d+) passed", output)
    m_fail = re.search(r"(\d+) failed", output)
    m_err = re.search(r"(\d+) error", output)
    n_pass = int(m_pass.group(1)) if m_pass else 0
    n_fail = int(m_fail.group(1)) if m_fail else 0
    n_err = int(m_err.group(1)) if m_err else 0
    if n_pass == 0 and n_fail == 0 and n_err == 0:
        return None
    return 1.0 if (n_fail == 0 and n_err == 0 and n_pass > 0) else 0.0


def tb_replay_phi(inst: dict, commands: list, deadline: float | None = None,
                  timeout_s: int = 1800) -> float | None:
    """Replay recorded bash commands in a fresh TB environment and score the final state.

    TB counterpart of step_certificates._replay_phi, used by the replay and perturbation tools.
    TB grading uses only the final container state (there is no patch), so replaying the
    commands and running the task's tests gives a score with the same meaning as an episode's.

    Returns 1.0/0.0 like score(), or None on infra failure (build, start or exec errors); None
    must never be read as evidence. Checks `deadline` (time.monotonic()) between commands so a
    probe cannot overrun its time budget."""
    import time as _time
    session = TBContainerSession(inst.get("instance_id", "?"), src=inst.get("tb_src", ""))
    try:
        if not session.start():
            return None
        for cmd in commands:
            if deadline is not None and _time.monotonic() > deadline:
                logger.warning(f"[tb_env] replay deadline hit ({session.instance_id})")
                return None
            if not (cmd or "").strip():
                continue
            session.execute(cmd)          # observations discarded: only the end state matters
        return session.score()
    except Exception as e:
        logger.warning(f"[tb_env] tb_replay_phi error ({inst.get('instance_id')}): {e}")
        return None
    finally:
        session.close()


def run_tb_task(inst: dict, system_prompt: str, url: str, model: str, *,
                max_turns: int = 100, bedrock_region: str = "",
                steps_out: list | None = None, hooks_src: str | None = None) -> tuple:
    """One executor episode on a Terminal-Bench task. Returns (phi, transcript).

    Same structure as executor.run_task (bash and submit tools, region rotation, sliding-window
    observation shrinking). The differences: a TB container session instead of a SWE one, and
    scoring by running the task's tests in the container at the end instead of grading a git
    diff. hooks_src adds the code_hooks executable layer as in run_task (None: no hooks)."""
    from verse.runtime.executor import (
        _invoke_bedrock_raw, _EXECUTOR_TOOLS, _shrink_old_observations, _KEEP_FULL_TURNS,
        _TEMP_DEPRECATED, truncate_observation)

    instance_id = inst.get("instance_id", "?")
    problem = (inst.get("problem_statement") or inst.get("instruction") or "")[:8000]
    session = TBContainerSession(instance_id, src=inst.get("tb_src", ""))
    hooks_rt, hooks_load_error = None, None
    if hooks_src:
        from verse.evolution.hooks_runtime import (
            HookedRuntime, HookLoadError, BashCapability, summary_transcript_entry)
        try:
            hooks_rt = HookedRuntime(hooks_src)
        except HookLoadError as e:
            hooks_load_error = str(e)[:600]
    if hooks_rt is not None:
        system_prompt = hooks_rt.system_prompt(system_prompt)
    transcript = [{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Task:\n{problem}"}]
    if hooks_load_error:
        transcript.append({"role": "system",
                           "content": f"HARNESS_CODE_LOAD_ERROR: {hooks_load_error}"})
    if url != "bedrock":
        return float("nan"), transcript          # TB supports only the tool-calling path
    region = bedrock_region or "us-west-2"
    deadline = time.monotonic() + float(os.environ.get("T2E_EPISODE_WALL_S", "2700"))
    # loop settings and the hook LLM budget, as in run_task (see executor.py)
    obs_cap, keep_full, spam_cap = 4096, _KEEP_FULL_TURNS, 6
    nudge_no_tool = ("You did not call a tool. Call `bash` to run a command, or `submit` "
                     "when the task is complete.")
    nudge_bad_markup = ("Your tool call did not parse — emit a NATIVE tool call (the bash/"
                        "submit tools), not XML or markup in your text.")
    tools = _EXECUTOR_TOOLS
    cap = None
    if hooks_rt is not None:
        lc = hooks_rt.loop_config(max_turns)
        max_turns = lc.get("max_turns", max_turns)
        obs_cap = lc.get("obs_cap", obs_cap)
        keep_full = lc.get("keep_full_turns", keep_full)
        spam_cap = lc.get("spam_streak", spam_cap)
        nudge_no_tool = lc.get("nudge_no_tool", nudge_no_tool)
        nudge_bad_markup = lc.get("nudge_bad_markup", nudge_bad_markup)
        def _hook_llm(prompt, mt):
            b = {"max_tokens": mt,
                 "messages": [{"role": "user", "content": prompt}]}
            if not any(t in model for t in _TEMP_DEPRECATED):
                b["temperature"] = 0.0
            r = _invoke_bedrock_raw(model, region, b)
            return "\n".join(x.get("text", "") for x in r.get("content", [])
                             if x.get("type") == "text")
        hooks_rt.attach_llm(_hook_llm, max_calls=max_turns // 2)
        extra_specs = hooks_rt.extra_tool_specs()
        if extra_specs:
            tools = _EXECUTOR_TOOLS + extra_specs
        cap = BashCapability(session.execute, deadline)
    extra_names = {t["name"] for t in tools} - {"bash", "submit"}
    msgs = [{"role": "user", "content": f"Task:\n{problem}"}]
    phi = None
    no_tool_streak = 0
    try:
        if not session.start():
            return float("nan"), transcript      # infra (build/start), not a verdict
        for turn in range(max_turns):
            if (hooks_rt is not None and hooks_rt.llm_cap is not None
                    and turn + hooks_rt.llm_cap.calls >= max_turns):
                transcript.append({"role": "user",
                                   "content": "(turn budget consumed: agent turns + "
                                              "hook llm calls hit the cap)"})
                break
            if time.monotonic() > deadline:
                transcript.append({"role": "user", "content": "(episode wall-clock cap hit)"})
                break
            if turn and turn % 8 == 0:
                _shrink_old_observations(msgs, keep_recent_msgs=keep_full * 2)
            body_msgs = hooks_rt.before_llm(msgs, turn) if hooks_rt is not None else msgs
            body = {"max_tokens": 4096,
                    "system": system_prompt, "tools": tools, "messages": body_msgs}
            if not any(t in model for t in _TEMP_DEPRECATED):
                body["temperature"] = 0.0
            resp = _invoke_bedrock_raw(model, region, body)
            content = resp.get("content", [])
            if hooks_rt is not None:
                content = hooks_rt.after_llm(content, turn)
            msgs.append({"role": "assistant", "content": content})
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            rec = {"turn": turn, "text": "\n".join(t for t in texts if t), "commands": [],
                   "observations": []}
            asst_lines = [t for t in texts if t.strip()]

            if not tool_uses:
                transcript.append({"role": "assistant",
                                   "content": "\n".join(asst_lines) or "(no action)"})
                from verse.runtime.executor import _looks_like_failed_toolcall
                fake_call = _looks_like_failed_toolcall(texts)
                if resp.get("stop_reason") == "end_turn" and turn > 0 and not fake_call:
                    if steps_out is not None:
                        rec["kind"] = "end_no_tool"; steps_out.append(rec)
                    break
                no_tool_streak += 1
                if no_tool_streak >= spam_cap:
                    transcript.append({"role": "user",
                                       "content": "(episode ended: repeated unparseable turns)"})
                    if steps_out is not None:
                        rec["kind"] = "spam_abort"; steps_out.append(rec)
                    break
                nudge = nudge_bad_markup if fake_call else nudge_no_tool
                msgs.append({"role": "user", "content": nudge})
                transcript.append({"role": "user", "content": nudge})
                if steps_out is not None:
                    rec["kind"] = "nudge"; steps_out.append(rec)
                continue

            tool_results = []
            submitted = False
            real_action = False       # any submit / extra tool / non-empty bash this turn
            for tu in tool_uses:
                name, tid = tu.get("name"), tu.get("id")
                if name == "submit":
                    submitted = True
                    real_action = True
                    tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                         "content": "Submitted."})
                    continue
                if name in extra_names and hooks_rt is not None:
                    real_action = True
                    n0 = len(cap.commands)
                    out = hooks_rt.run_tool(name, tu.get("input") or {}, cap, turn)
                    out_t = truncate_observation(out, obs_cap)
                    tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                         "content": out_t})
                    for c in cap.commands[n0:]:
                        rec["commands"].append(c)
                        asst_lines.append(f"```bash\n{c}\n```")
                    rec["observations"].append(f"[tool {name}] " + out_t[:1000])
                    continue
                t_args = tu.get("input") or {}
                if hooks_rt is not None:
                    t_args, block, synth = hooks_rt.before_tool(name or "bash", t_args, turn)
                    if block:
                        real_action = True
                        obs = truncate_observation(synth, obs_cap) if synth \
                            else f"[harness] command blocked: {block}"
                        tool_results.append({"type": "tool_result", "tool_use_id": tid,
                                             "content": obs})
                        rec["observations"].append(obs[:1000])
                        asst_lines.append(
                            f"(synthetic: {t_args.get('command', '')[:80]})" if synth
                            else f"(blocked: {t_args.get('command', '')[:80]})")
                        continue
                cmd = t_args.get("command", "")
                if cmd.strip():
                    real_action = True
                    obs = session.execute(cmd)
                else:
                    # Empty-command guard: weak models sometimes emit a bash tool_use with an
                    # empty `command` when trying to write a large here-doc, then repeat it.
                    # Count it toward the spam streak (below) and return an actionable error.
                    obs = ("ERROR: the bash tool_use arrived with an EMPTY `command` "
                           "argument (this happens when a large file body is attempted "
                           "in one call). Retry with a real command; write big files in "
                           "PIECES: printf '...' > file, then printf '...' >> file, or "
                           "several small heredocs.")
                if hooks_rt is not None:
                    # execute() output is already bounded, but a hook may return a longer
                    # rewrite: cap it after the hook, as run_task does, so one faulty
                    # after_tool cannot flood the context of later turns
                    obs = truncate_observation(
                        hooks_rt.after_tool(name or "bash", t_args, obs, turn), obs_cap)
                rec["commands"].append(cmd)
                rec["observations"].append(obs)
                # Render as a ```bash``` fence, not "$ cmd": transcript_steps re-parses the
                # transcript with parse_action, which only recognizes fenced blocks, so
                # attribution and the verification tool can re-extract these commands for replay.
                asst_lines.append(f"```bash\n{cmd}\n```")
                tool_results.append({"type": "tool_result", "tool_use_id": tid, "content": obs})
            # empty-command turns count toward the spam guard (a tool_use with a blank command
            # is not a real action); real actions reset it
            if real_action:
                no_tool_streak = 0
            else:
                no_tool_streak += 1
                if no_tool_streak >= spam_cap:
                    transcript.append({"role": "assistant",
                                       "content": "\n".join(asst_lines) or "(tool call)"})
                    transcript.append({"role": "user",
                                       "content": "(episode ended: repeated empty commands)"})
                    if steps_out is not None:
                        rec["kind"] = "spam_abort"; steps_out.append(rec)
                    break
            transcript.append({"role": "assistant", "content": "\n".join(asst_lines) or "(tool call)"})
            # Observations also go into the readable transcript (as in run_task), so the
            # optimizer sees what the agent saw when it diagnoses failures.
            if rec["observations"]:
                transcript.append({"role": "user",
                                   "content": "\n---\n".join(
                                       o[:2000] for o in rec["observations"])})
            if hooks_rt is not None and not submitted:
                note = hooks_rt.on_turn_end(turn)
                if note:
                    tool_results.append({"type": "text", "text": f"[harness note] {note}"})
                    transcript.append({"role": "user", "content": f"[harness note] {note}"})
            msgs.append({"role": "user", "content": tool_results})
            if steps_out is not None:
                rec["kind"] = "act"; steps_out.append(rec)
            if submitted:
                break
        # score the final container state via the task's own tests
        phi = session.score(run_id=session.name)
        # On failure, the transcript's last line names the failed tests. Attribution minimizes
        # toward this failure signature: TB failures are usually omissions, so the agent's own
        # observations show no error, but the test verdict does.
        if phi is not None and phi < 0.5 and session.last_test_output:
            # The grader-log tail (assert details, not just test names; see run_task) goes
            # before the one-line marker, which must stay the transcript's last line for the
            # verification tool and attribution.
            from verse.runtime.executor import _grader_tail_entry
            tail = _grader_tail_entry(session.last_test_output)
            if tail:
                transcript.append(tail)
            fails = re.findall(r"(?:FAILED|ERROR)\s+(\S+)", session.last_test_output)
            marker = ("TEST FAILURES: " + " ".join(sorted(set(fails))[:8])) if fails else \
                     ("TEST RESULT: " + session.last_test_output.strip().splitlines()[-1][:160]
                      if session.last_test_output.strip() else "TEST RESULT: task not resolved")
            transcript.append({"role": "user", "content": marker})
    except Exception as e:
        logger.warning(f"[tb_env] run_tb_task exception ({instance_id}): {e}")
        phi = None
    finally:
        session.close()
        if hooks_rt is not None:
            transcript.append(summary_transcript_entry(hooks_rt))
    if phi is None:
        return float("nan"), transcript
    return float(phi), transcript
