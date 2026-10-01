"""Container-backed shell environment for the SWE executor.

SWEContainerSession keeps one long-lived docker container per trajectory, started from the task
image (repository pre-installed at /testbed). execute(command) runs a shell command in it with
`docker exec`, so file edits persist across turns, as in SWE-agent's bash session. At the end,
get_patch() returns the candidate patch (`git diff` of /testbed). This module does no scoring:
swe_judge grades the patch in a separate, fresh container from the same image.

SWEChrootSession is a dockerless equivalent (see chroot_exec.py); make_session picks the backend.
Failures do not raise: execute returns an error string and get_patch returns "".

Environment variables:
    T2E_SWE_DOCKER            docker command (default "docker"; e.g. "sudo docker")
    T2E_SWE_EXEC_BACKEND      "docker" (default) or "chroot"
    T2E_SWE_EXEC_TIMEOUT      per-command timeout in seconds (default 120)
    T2E_SWE_START_TIMEOUT     container start timeout in seconds (default 300)
    T2E_SWE_WORKDIR           container working directory (default /testbed)
    T2E_SWE_CONTAINER_CPUS    CPU limit per container (default 6)
    T2E_SWE_DOCKER_RUN_ARGS   extra `docker run` flags
    T2E_SWE_ALLOW_NET         "1" gives rollout containers network access (default offline)
    T2E_SWE_SCRUB_GIT         "1" removes the repository's git history before the episode
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import uuid

logger = logging.getLogger("verse.runtime")

_WORKDIR = os.environ.get("T2E_SWE_WORKDIR", "/testbed")


# Action parsing for text-mode executors (the tool-calling executor does not use it).
# Commands come as fenced bash blocks (SWE-agent / mini-SWE-agent convention):
#   ```bash
#   sed -i 's/foo/bar/' src/x.py
#   ```
# or as <bash> tags, with several commands allowed per turn (SWE-agent's all_bash_code_blocks
# convention). A fenced ```diff block is a direct patch submission, and a standalone "submit" or
# "done" line ends the episode. The closing fence must be on its own line, so a ``` inside the
# body does not end the block.
_BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)\n```", re.DOTALL)
_BASH_TAG_RE = re.compile(r"<bash>\s*\n?(.*?)\n?\s*</bash>", re.DOTALL)
# accept both `diff --git` and a bare unified diff (`--- a/... / +++ b/...`) inside the fence
_DIFF_BLOCK_RE = re.compile(
    r"```(?:diff|patch)?[ \t]*\n((?:diff --git |--- (?:a/|/dev/null)).*?)\n```", re.DOTALL)
# submit must be a standalone line, so prose such as "once I'm done..." does not end the episode
_SUBMIT_RE = re.compile(r"^\s*(?:submit|done|finished|<<\s*submit\s*>>)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_action(assistant_text: str) -> dict:
    """Parse one text-executor assistant turn into an action dict the loop dispatches on.

    Returns one of:
        {"kind": "patch",  "patch": <unified diff str>}                 # diff submission; ends
        {"kind": "bash",   "command": <str>, "commands": [<str>, ...]}  # run, return output
        {"kind": "submit"}                                              # end; use the repo diff
        {"kind": "noop"}                                                # nothing to act on
    Precedence: a diff block wins; else bash blocks or tags; else submit.
    The tool-calling executor (executor.run_task) does not use this parser.
    """
    if not assistant_text:
        return {"kind": "noop"}
    mdiff = _DIFF_BLOCK_RE.search(assistant_text)
    if mdiff:
        return {"kind": "patch", "patch": mdiff.group(1)}
    # collect all commands from ```bash fences and <bash> tags
    cmds = [c.strip() for c in _BASH_BLOCK_RE.findall(assistant_text) if c.strip()]
    cmds += [c.strip() for c in _BASH_TAG_RE.findall(assistant_text) if c.strip()]
    if cmds:
        return {"kind": "bash", "command": cmds[0], "commands": cmds}
    if _SUBMIT_RE.search(assistant_text):
        return {"kind": "submit"}
    return {"kind": "noop"}


def truncate_observation(text: str, max_chars: int = 4000) -> str:
    """Truncate a tool observation to max_chars, keeping the head and tail (as SWE-agent does)."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n... [truncated {len(text) - max_chars} chars] ...\n{tail}"


# Leak guard: each live session registers its container name here, and an atexit hook removes
# any containers still open when the process exits (including after an exception that skips
# close()). close() still removes each container right away.
_LIVE_CONTAINERS: set[str] = set()
_ATEXIT_REGISTERED = False


def _register_leak_guard():
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    _ATEXIT_REGISTERED = True
    import atexit

    def _sweep():
        if not _LIVE_CONTAINERS:
            return
        docker = os.environ.get("T2E_SWE_DOCKER", "docker")
        names = " ".join(shlex.quote(n) for n in list(_LIVE_CONTAINERS))
        try:
            subprocess.run(f"{docker} rm -f {names}", shell=True, capture_output=True, timeout=120)
        except Exception:
            pass
    atexit.register(_sweep)


# Sessions.
class SWEContainerSession:
    """A persistent docker container for one SWE trajectory.

    Lifecycle: start, execute (repeatedly), get_patch, close. The docker command comes from
    T2E_SWE_DOCKER (default "docker"). Failures return False, an error string or "" instead of
    raising."""

    def __init__(self, image: str, name: str | None = None):
        self.image = image
        self.name = name or f"t2e_swe_{uuid.uuid4().hex[:12]}"
        self.docker = os.environ.get("T2E_SWE_DOCKER", "docker")
        self.exec_timeout = int(os.environ.get("T2E_SWE_EXEC_TIMEOUT", "120"))
        self._started = False

    def _run(self, args: str, timeout: int) -> subprocess.CompletedProcess:
        """Run a docker command in its own process group and kill the group on timeout.

        subprocess.run(timeout=...) cannot kill `sudo docker ...`: the child is owned by root,
        the unprivileged kill fails, and the following wait() blocks indefinitely, which can
        stall a whole thread pool. Starting a new session and killing its process group (with
        `sudo -n kill` when available) avoids this."""
        cmd = f"{self.docker} {args}"
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

    def start(self) -> bool:
        """Start the container detached (running `sleep`) so commands can run via `docker exec`."""
        try:
            # Network isolation (default on): the task image has all dependencies installed
            # (pytest runs with --network none), so the agent needs no internet. With network
            # access, an evolved harness can clone the upstream repository and copy the reference
            # fix; public SWE-bench evaluations also run rollouts offline. T2E_SWE_ALLOW_NET=1
            # enables the network.
            net = "" if os.environ.get("T2E_SWE_ALLOW_NET") == "1" else "--network none"
            # --cpus cap: some test suites run parallel workers (pytest-xdist), and an uncapped
            # container can take most of the host's cores and starve the other containers
            cpus = os.environ.get("T2E_SWE_CONTAINER_CPUS", "6")
            # extra `docker run` flags shared with swe_judge (e.g. the IPv6 loopback sysctl), so
            # rollout and eval containers see the same kernel settings
            extra = os.environ.get("T2E_SWE_DOCKER_RUN_ARGS", "").strip()
            # the start timeout also covers an implicit image pull when the image is not cached;
            # a cold multi-GB pull on a slow disk can exceed 300s, so raise it on such hosts
            start_to = int(os.environ.get("T2E_SWE_START_TIMEOUT", "300"))
            cp = self._run(
                f"run -d --name {shlex.quote(self.name)} {net} {extra} --cpus={cpus} "
                f"-w {shlex.quote(_WORKDIR)} "
                f"{shlex.quote(self.image)} sleep 86400",
                timeout=start_to,
            )
            if cp.returncode != 0:
                logger.warning(f"[swe_env] container start failed: {cp.stderr[-300:]}")
                # remove a partially created container by name (close() skips it because
                # _started is False)
                try:
                    self._run(f"rm -f {shlex.quote(self.name)}", timeout=60)
                except Exception:
                    pass
                return False
            # Harbor-packaged SWE-rebench V2 images keep the repo at /<repo-name>. Link /testbed to
            # the top-level git checkout so the /testbed-based prompt, tool description and patch
            # extraction work unchanged. No-op when /testbed/.git already exists.
            self._run(f"exec {shlex.quote(self.name)} bash -lc " + shlex.quote(
                      "[ -d " + _WORKDIR + "/.git ] || { r=$(ls -d /*/.git 2>/dev/null | grep -v '^" + _WORKDIR
                      + "/' | head -1); [ -n \"$r\" ] && { rmdir " + _WORKDIR + " 2>/dev/null; "
                      "ln -sfn \"$(dirname \"$r\")\" " + _WORKDIR + "; }; }; true"), timeout=60)
            # mark the repo as a git safe.directory so git commands (and get_patch) work in it
            self._run(f"exec {shlex.quote(self.name)} bash -lc "
                      f"{shlex.quote('cd '+_WORKDIR+' && git config --global --add safe.directory '+_WORKDIR+' || true')}",
                      timeout=60)
            # T2E_SWE_SCRUB_GIT=1: remove the git history in the rollout container so the agent
            # cannot find the upstream fix commit in the image's .git. The repo is re-initialized
            # with one baseline commit so get_patch() (`git add -N`, `git diff HEAD`) still works.
            # The eval container (swe_judge) is a separate fresh container with the full history
            # for `git reset --hard <base_commit>`.
            if os.environ.get("T2E_SWE_SCRUB_GIT", "0") == "1":
                scrub = ("cd " + _WORKDIR + " && rm -rf .git && git init -q && "
                         "git config user.email t2e@local && git config user.name t2e && "
                         "git add -A >/dev/null 2>&1; git commit -qm baseline >/dev/null 2>&1 || true")
                cp2 = self._run(f"exec {shlex.quote(self.name)} bash -lc {shlex.quote(scrub)}",
                                timeout=180)
                if cp2.returncode != 0:
                    logger.warning(f"[swe_env] git scrub failed ({self.image}): {cp2.stderr[-200:]}")
                    self._run(f"rm -f {shlex.quote(self.name)}", timeout=60)
                    return False   # do not run unscrubbed when scrubbing was requested
            self._started = True
            _register_leak_guard()
            _LIVE_CONTAINERS.add(self.name)   # removed at exit if close() never runs
            return True
        except Exception as e:
            logger.warning(f"[swe_env] container start exception: {e}")
            return False

    def execute(self, command: str) -> str:
        """Run a shell command in the working directory; return truncated stdout and stderr.

        The command runs through `bash -lc`, so the image profile activates the conda `testbed`
        env."""
        if not self._started:
            return "ERROR: container not started"
        wrapped = f"cd {_WORKDIR} && {command}"
        try:
            cp = self._run(
                f"exec {shlex.quote(self.name)} bash -lc {shlex.quote(wrapped)}",
                timeout=self.exec_timeout,
            )
            out = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
            return truncate_observation(out)
        except subprocess.TimeoutExpired:
            # return guidance rather than a bare error: a full test suite can exceed the
            # per-command limit, and after a bare error executors tend to repeat the same command
            return (f"ERROR: command timed out after {self.exec_timeout}s. Do not re-run "
                    "the same command — narrow it (single test file or -k pattern, e.g. "
                    "`python -m pytest path/to/test_one.py -x -q`) so it fits the limit.")
        except Exception as e:
            return f"ERROR: {e}"

    def get_patch(self) -> str:
        """Return the candidate patch: `git diff HEAD` of the repo, minus build and cache files.

        No `git add -A`: it would stage the agent's scratch files (temp scripts, downloaded data,
        __pycache__), and the diff could then fail to apply in the grader. Edits to test files
        are kept; swe_judge applies this patch first and then restores the test files and
        applies the task's test patch, which overrides any change to the graded tests."""
        if not self._started:
            return ""
        # exclude build, cache and VCS files; keep source and test edits (the grader restores tests)
        excludes = (
            " ':(exclude)*.pyc' ':(exclude)**/__pycache__/**' ':(exclude).git/**' "
            "':(exclude)*.egg-info/**' ':(exclude)**/*.so' ':(exclude).pytest_cache/**' "
            "':(exclude)build/**' ':(exclude)dist/**'"
        )
        # `git add -N .` marks new files as intent-to-add, so `git diff HEAD` shows them as
        # additions next to the modified tracked files; the exclusions drop the scratch files.
        script = (
            "cd " + _WORKDIR + " && "
            "git add -N . >/dev/null 2>&1 || true; "
            "git diff HEAD -- . " + excludes
        )
        try:
            cp = self._run(
                f"exec {shlex.quote(self.name)} bash -lc {shlex.quote(script)}",
                timeout=max(self.exec_timeout, 120),   # honor the configured timeout (large diffs)
            )
            return cp.stdout or ""
        except Exception as e:
            logger.warning(f"[swe_env] get_patch failed: {e}")
            return ""

    def close(self) -> None:
        if not self._started:
            return
        try:
            self._run(f"rm -f {shlex.quote(self.name)}", timeout=60)
        except Exception:
            pass
        _LIVE_CONTAINERS.discard(self.name)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


class SWEChrootSession:
    """Dockerless equivalent of SWEContainerSession (root and chroot, no docker daemon).

    Each session hardlink-clones the shared extracted rootfs (chroot_exec.ensure_rootfs) with
    `cp -al` (seconds, almost no extra disk), except /testbed, which is a real copy: the agent
    edits files in place (>>, sed -i), and hardlinks would share those edits with other sessions
    of the same task. /tmp in the clone starts empty. close() removes the clone.

    Same public interface as SWEContainerSession: start, execute, get_patch, close; failures
    return error strings instead of raising."""

    def __init__(self, image: str, name: str | None = None):
        self.image = image
        self.name = name or f"t2e_chroot_{uuid.uuid4().hex[:12]}"
        self.exec_timeout = int(os.environ.get("T2E_SWE_EXEC_TIMEOUT", "120"))
        self.root = ""
        self._started = False

    def start(self) -> bool:
        from verse.runtime.chroot_exec import ensure_rootfs, _ensure_dev_nodes
        base = ensure_rootfs(self.image)
        if base is None:
            return False
        clone_root = os.path.join(os.path.dirname(base), "_sessions")
        self.root = os.path.join(clone_root, self.name)
        try:
            os.makedirs(clone_root, exist_ok=True)
            # hardlink-clone the OS, real-copy the repo the agent edits
            subprocess.run(["cp", "-al", base, self.root], check=True, capture_output=True,
                           timeout=300)
            tb = os.path.join(self.root, _WORKDIR.lstrip("/"))
            subprocess.run(["rm", "-rf", tb], check=True, capture_output=True, timeout=120)
            subprocess.run(["cp", "-a", os.path.join(base, _WORKDIR.lstrip("/")), tb],
                           check=True, capture_output=True, timeout=600)
            tmp = os.path.join(self.root, "tmp")
            subprocess.run(["rm", "-rf", tmp], capture_output=True, timeout=60)
            os.makedirs(tmp, exist_ok=True)
            # 0o700 instead of the usual 1777: every chroot command runs as root (chroot
            # requires it), so a world-writable /tmp would add attack surface and no function
            os.chmod(tmp, 0o700)
            _ensure_dev_nodes(self.root)
            for etc in ("resolv.conf", "hosts"):
                try:
                    import shutil as _sh
                    _sh.copyfile(f"/etc/{etc}", os.path.join(self.root, "etc", etc))
                except Exception:
                    pass
            self._chroot(f"cd {_WORKDIR} && git config --global --add safe.directory {_WORKDIR} || true",
                         timeout=60)
            # same git-history scrub as the docker session (see SWEContainerSession.start)
            if os.environ.get("T2E_SWE_SCRUB_GIT", "0") == "1":
                cp2 = self._chroot(
                    "cd " + _WORKDIR + " && rm -rf .git && git init -q && "
                    "git config user.email t2e@local && git config user.name t2e && "
                    "git add -A >/dev/null 2>&1; git commit -qm baseline >/dev/null 2>&1 || true",
                    timeout=180)
                if cp2.returncode != 0:
                    logger.warning(f"[swe_env] chroot git scrub failed: {cp2.stderr[-200:]}")
                    subprocess.run(["rm", "-rf", self.root], capture_output=True, timeout=120)
                    return False
            self._started = True
            return True
        except Exception as e:
            logger.warning(f"[swe_env] chroot session start failed: {e}")
            subprocess.run(["rm", "-rf", self.root], capture_output=True, timeout=120)
            return False

    def _chroot(self, script: str, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(["chroot", self.root, "/bin/bash", "-lc", script],
                              capture_output=True, text=True, timeout=timeout)

    def execute(self, command: str) -> str:
        if not self._started:
            return "ERROR: session not started"
        try:
            cp = self._chroot(f"cd {_WORKDIR} && {command}", timeout=self.exec_timeout)
            out = (cp.stdout or "") + (("\n" + cp.stderr) if cp.stderr else "")
            return truncate_observation(out)
        except subprocess.TimeoutExpired:
            # return guidance rather than a bare error: a full test suite can exceed the
            # per-command limit, and after a bare error executors tend to repeat the same command
            return (f"ERROR: command timed out after {self.exec_timeout}s. Do not re-run "
                    "the same command — narrow it (single test file or -k pattern, e.g. "
                    "`python -m pytest path/to/test_one.py -x -q`) so it fits the limit.")
        except Exception as e:
            return f"ERROR: {e}"

    def get_patch(self) -> str:
        """Same tracked-diff recipe as the docker session (see SWEContainerSession.get_patch)."""
        if not self._started:
            return ""
        excludes = (
            " ':(exclude)*.pyc' ':(exclude)**/__pycache__/**' ':(exclude).git/**' "
            "':(exclude)*.egg-info/**' ':(exclude)**/*.so' ':(exclude).pytest_cache/**' "
            "':(exclude)build/**' ':(exclude)dist/**'"
        )
        script = ("cd " + _WORKDIR + " && git add -N . >/dev/null 2>&1 || true; "
                  "git diff HEAD -- . " + excludes)
        try:
            cp = self._chroot(script, timeout=max(self.exec_timeout, 120))
            return cp.stdout or ""
        except Exception as e:
            logger.warning(f"[swe_env] chroot get_patch failed: {e}")
            return ""

    def close(self) -> None:
        if not self._started:
            return
        subprocess.run(["rm", "-rf", self.root], capture_output=True, timeout=300)
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def make_session(image: str, name: str | None = None):
    """Return a session for `image` from the backend that T2E_SWE_EXEC_BACKEND selects.

    SWEChrootSession for "chroot", else SWEContainerSession. Call sites use this so the backend
    is set by the environment alone."""
    if os.environ.get("T2E_SWE_EXEC_BACKEND", "docker") == "chroot":
        return SWEChrootSession(image, name)
    return SWEContainerSession(image, name)
