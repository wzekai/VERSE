"""Grade SWE tasks by execution: apply a candidate patch, run the task's tests, return 1 or 0.

A task is resolved when, with the patch applied, every FAIL_TO_PASS test passes and every
PASS_TO_PASS test still passes (the SWE-bench resolution criterion).

SWE-rebench rows. SWE-rebench covers hundreds of repositories, and swebench's per-repo specs
(make_test_spec, MAP_REPO_TO_PARSER) know only the original SWE-bench repositories. Each
SWE-rebench image instead ships the repository pre-installed at /testbed in a conda `testbed`
env, so one generic recipe works for every task:
    1. reset /testbed to base_commit and apply the candidate patch;
    2. restore the files touched by test_patch, then apply test_patch;
    3. run the task's test command on the FAIL_TO_PASS and PASS_TO_PASS node ids;
    4. parse the per-test PASSED/FAILED lines from the log.
swebench's log parser is used when the repository is registered there; otherwise a generic
pytest `-rA` parser is used.

Harbor rows (rows with a `verifier_dir`, written by scripts/swe_harbor_import.py). The task
ships its own verifier (tests/test.sh), which applies the test patch, runs and parses the tests
and writes /logs/verifier/reward.txt. The judge applies the candidate patch, runs the verifier
and reads the reward.

The candidate patch comes from `extra_info['model_patch']` or, as a fallback, is parsed from
`solution_str`. The task fields (instance_id, repo, base_commit, test_patch, FAIL_TO_PASS,
PASS_TO_PASS, image, ...) come from `extra_info`.

A run that cannot be executed (docker failure, timeout, test patch that does not apply) is an
infrastructure failure, not an unresolved task; see INFRA_SENTINEL. The module imports without
swebench or docker installed.

Environment variables:
    T2E_SWE_DOCKER            docker command (default "docker"; e.g. "sudo docker")
    T2E_SWE_EXEC_BACKEND      "docker" (default) or "chroot" (see chroot_exec.py)
    T2E_SWE_IMAGE_PREFIX      registry that replaces a "<your-docker-registry>/" placeholder in a
                              row's image name (default "docker.io/")
    T2E_SWE_ECR_PREFIX        optional registry mirror for all task images (see _image_name)
    T2E_SWE_DOCKER_RUN_ARGS   extra `docker run` flags
    T2E_SWE_CONTAINER_CPUS    CPU limit per eval container (default 6)
    T2E_SWE_JUDGE_ALLOW_NET   "1" gives the eval container network access (default offline)
    T2E_SWE_RUN_TIMEOUT       per-task test timeout in seconds (default 1800)
    T2E_SWE_KEEP_CONTAINER    "1" keeps the eval container for debugging (default: removed)
    T2E_SWE_INFRA_AS_ZERO     "1" scores infrastructure failures 0.0 instead of NaN
    T2E_SWE_INFRA_LOG         JSONL file that records every infrastructure failure
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile

logger = logging.getLogger("verse.runtime")

# Returned by compute_score when a run could not be executed (docker or image failure, timeout,
# or a test patch that does not apply), as opposed to an unresolved task. swe_bc_aggregate drops
# these from the denominator. T2E_SWE_INFRA_AS_ZERO=1 returns 0.0 instead, for callers that need
# a numeric score.
INFRA_SENTINEL = float("nan")


def swe_bc_aggregate(scores) -> dict:
    """Aggregate compute_score outputs into a resolve rate over valid executions.

    Infrastructure failures (NaN) are excluded from the denominator:
        bc = (# resolved) / (# non-NaN scores)
    Returns {bc, bc_pct, n_total, n_valid, n_infra, n_resolved}. A plain mean would count
    infrastructure failures as unresolved."""
    import math
    scores = list(scores)
    n_total = len(scores)
    valid = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
    n_valid = len(valid)
    n_infra = n_total - n_valid
    n_resolved = sum(1 for s in valid if s >= 0.5)
    bc = (n_resolved / n_valid) if n_valid else 0.0
    return {"bc": bc, "bc_pct": round(bc * 100, 2), "n_total": n_total,
            "n_valid": n_valid, "n_infra": n_infra, "n_resolved": n_resolved}


# Patch extraction.
# Both `diff --git` diffs and bare `--- a/<f>` / `+++ b/<f>` unified diffs are accepted.
_DIFF_GIT_RE = re.compile(r"(diff --git .*?)(?=\Z)", re.DOTALL)
_DIFF_UNIFIED_RE = re.compile(r"(^--- (?:a/|/dev/null).*?)(?=\Z)", re.DOTALL | re.MULTILINE)
_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)


def _looks_like_diff(s: str) -> bool:
    return bool(s) and ("diff --git" in s or re.search(r"^--- (?:a/|/dev/null)", s, re.MULTILINE) is not None)


def _extract_patch(extra_info, solution_str) -> str:
    """Return the candidate patch as a unified diff, or "" if there is none.

    Prefers the patch in extra_info (the container's own `git diff`). Otherwise scans
    solution_str: first a fenced ```diff block (so prose after the block is not captured), then
    a raw `diff --git` block, then a bare `--- a/ ... +++ b/` block."""
    if isinstance(extra_info, dict):
        for k in ("model_patch", "patch", "prediction_patch"):
            v = extra_info.get(k)
            if isinstance(v, str) and _looks_like_diff(v):
                return v
    text = solution_str or ""
    # 1. a fenced ```diff / ```patch block
    for fm in _FENCE_RE.finditer(text):
        body = fm.group(1)
        if _looks_like_diff(body):
            return body.strip() + "\n"
    # 2. a raw `diff --git ...` block to end-of-text
    m = _DIFF_GIT_RE.search(text)
    if m:
        return m.group(1)
    # 3. a raw bare-unified-diff (`--- a/... / +++ b/...`) block to end-of-text
    m = _DIFF_UNIFIED_RE.search(text)
    if m:
        return m.group(1)
    return ""


# Task assembly.
_LIST_FIELDS = ("FAIL_TO_PASS", "PASS_TO_PASS")


def _as_instance(extra_info) -> dict | None:
    """Build the task dict from a row's extra_info, or return None if required fields are missing.

    FAIL_TO_PASS and PASS_TO_PASS may be stored as JSON strings; they are normalized to lists."""
    if not isinstance(extra_info, dict):
        return None
    needed = ("instance_id", "repo", "base_commit", "test_patch")
    if not all(k in extra_info for k in needed):
        # the fields may be nested under 'swe_instance'
        inner = extra_info.get("swe_instance")
        if isinstance(inner, dict):
            extra_info = inner
        else:
            return None
    inst = dict(extra_info)
    for k in _LIST_FIELDS:
        v = inst.get(k)
        if isinstance(v, str):
            try:
                inst[k] = json.loads(v)
            except Exception:
                inst[k] = [v] if v else []
        elif v is None:
            inst[k] = []
    return inst


def _image_name(inst: dict) -> str:
    """Return the docker image for a task.

    Uses the row's docker_image or image_name as is (SWE-rebench rows carry a public ref such as
    'swerebench/sweb.eval.x86_64.<id>'). A leading '<...>/' registry placeholder is replaced by
    T2E_SWE_IMAGE_PREFIX. Without an image field, the name is derived from instance_id. When
    T2E_SWE_ECR_PREFIX is set (e.g. '<registry>/<prefix>'), every image is moved onto that
    registry mirror, keeping only the last path component (Docker Hub throttles anonymous
    pulls)."""
    img = inst.get("docker_image") or inst.get("image_name") or ""
    if img:
        if img.startswith("<"):
            img = re.sub(r"^<[^>]*>/", os.environ.get("T2E_SWE_IMAGE_PREFIX", "docker.io/"), img)
    else:
        iid = inst.get("instance_id", "")
        img = f"swerebench/sweb.eval.x86_64.{iid}"
    ecr = os.environ.get("T2E_SWE_ECR_PREFIX", "").rstrip("/")
    if ecr:
        img = f"{ecr}/{img.split('/')[-1]}"
    return img


# Generic SWE-rebench eval recipe.
# Markers printed around the test run so the parser can find the test output in the log.
_BEGIN = ">>>>> Start Test Output"
_END = ">>>>> End Test Output"


def _test_files_from_patch(test_patch: str) -> tuple[list[str], list[str]]:
    """Return (modified_files, new_files) touched by test_patch.

    The split follows the official SWE-bench eval script (swebench test_spec/python.py): files
    that exist at the base commit are reset with `git checkout`, and files the test patch
    creates are removed with `rm -f`. `git checkout <base> -- <new_file>` fails for a file that
    does not exist at base, and a same-path file left by the candidate patch would make the
    test patch's `new file` hunk fail to apply."""
    modified, new = [], []
    for block in re.split(r"^diff --git ", test_patch or "", flags=re.MULTILINE)[1:]:
        m = re.match(r"a/(\S+) b/(\S+)", block)
        if not m:
            continue
        is_new = bool(re.search(r"^(new file mode|--- /dev/null)", block, re.MULTILINE))
        (new if is_new else modified).append(m.group(2))
    return modified, new


def _test_cmd(inst: dict) -> str:
    """Return the task's test command.

    SWE-rebench rows carry install_config.test_cmd (some repositories, e.g. Django and SymPy,
    use a non-pytest runner). Falls back to `python -m pytest -rA` when it is absent or empty.
    install_config may be a JSON string or a dict; test_cmd may be a string or a list (the last
    entry is the runner)."""
    ic = inst.get("install_config")
    if isinstance(ic, str):
        try:
            ic = json.loads(ic)
        except Exception:
            ic = None
    if isinstance(ic, dict):
        tc = ic.get("test_cmd")
        if isinstance(tc, (list, tuple)) and tc:
            tc = tc[-1]
        if isinstance(tc, str) and tc.strip():
            return tc.strip()
    return "python -m pytest -rA --tb=no -p no:cacheprovider"


# Harbor-packaged SWE-rebench tasks.
# Tasks from the SWE-rebench V2 pipeline (e.g. the swe-rebench-07-2026 snapshot) ship as Harbor
# packages: tests/config.json (the SWE-bench-style record) and tests/test.sh, which applies the
# test patch, runs install_config.test_cmd, parses the log with tests/swan_log_parsers.py and
# writes /logs/verifier/reward.txt (1 or 0). The repository is at /<repo-name>, not /testbed.
# scripts/swe_harbor_import.py adds verifier_dir (host path of tests/), repo_dir and a locally
# built docker_image to each row. The judge keeps the same apply order but lets the task's own
# verifier run and parse the tests, as the SWE-rebench leaderboard does. _run_in_docker reads
# the marker line below to find the host directory to copy and mount at /tests (test.sh
# hardcodes that path).
_HARBOR_MARK = "# T2E_VERIFIER_DIR="


def _harbor_verifier_env(inst: dict) -> dict:
    """Return the [verifier.env] table of the task's task.toml (next to its tests/ directory).

    Harbor exports these variables into the verifier's shell. Some Java tasks need them: their
    images set preferIPv6Addresses=true, and in a container without IPv6 every Maven download
    fails with 'Network is unreachable' unless
    _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false is set. Returns {} when the file is
    missing or unreadable; the eval script then has no export lines."""
    vdir = str(inst.get("verifier_dir") or "").rstrip("/")
    if not vdir:
        return {}
    try:
        import tomllib
        with open(os.path.join(os.path.dirname(vdir), "task.toml"), "rb") as f:
            cfg = tomllib.load(f)
        env = (cfg.get("verifier") or {}).get("env") or {}
        return {str(k): str(v) for k, v in env.items()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k))}
    except Exception:
        return {}


def _build_harbor_eval_script(inst: dict) -> str:
    base = inst.get("base_commit", "")
    ver_env = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in _harbor_verifier_env(inst).items())
    repo_dir = inst.get("repo_dir") or ("/" + str(inst.get("repo", "x/x")).split("/")[-1])
    mod_files, new_files = _test_files_from_patch(inst.get("test_patch", ""))
    reset_cmds = []
    if mod_files:
        reset_cmds.append(f"git checkout {shlex.quote(base)} -- "
                          + " ".join(shlex.quote(f) for f in mod_files))
    if new_files:
        reset_cmds.append("rm -f " + " ".join(shlex.quote(f) for f in new_files))
    reset_tests = "\n".join(reset_cmds) if reset_cmds else "true"
    q = shlex.quote(repo_dir)
    return (
        "#!/bin/bash\n"
        f"{_HARBOR_MARK}{inst['verifier_dir']}\n"
        "set -uxo pipefail\n"
        # link /testbed to the repo dir (no-op when the image already has /testbed)
        f"[ -d /testbed/.git ] || {{ rmdir /testbed 2>/dev/null; ln -sfn {q} /testbed; }}\n"
        f"cd {q}\n"
        f"git config --global --add safe.directory {q}; git config --global --add safe.directory /testbed\n"
        # 1. reset to base  2. apply the candidate patch  3. reset the test files (test.sh applies
        #    the test patch last)
        f"git reset --hard {shlex.quote(base)} 2>/dev/null || git reset --hard\n"
        "git apply -v /t2e/model.patch || git apply --3way -v /t2e/model.patch || "
        "patch -p1 --fuzz=5 -i /t2e/model.patch || echo T2E_MODEL_PATCH_FAILED\n"
        f"{reset_tests}\n"
        "mkdir -p /logs/verifier; rm -f /logs/verifier/reward.txt /logs/verifier/report.json\n"
        f"{ver_env}"   # task.toml [verifier.env], as Harbor exports it to the verifier shell
        f"echo '{_BEGIN}'\n"
        "bash /tests/test.sh; echo \"T2E_HARBOR_EXIT=$?\"\n"
        "echo \"T2E_HARBOR_REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo none)\"\n"
        # no reward file means the verifier never reached its parser (the test patch did not apply,
        # or the verifier crashed): an infrastructure failure, flagged with the same marker as the
        # generic recipe
        "[ -f /logs/verifier/reward.txt ] || echo T2E_TEST_PATCH_FAILED\n"
        # the verifier (root inside the container) writes parser.py and __pycache__ into the
        # bind-mounted per-run copy of tests/; remove them so the host user can delete the temp dir
        "rm -rf /tests/__pycache__ /tests/parser.py 2>/dev/null; chmod -R a+rwX /tests 2>/dev/null || true\n"
        f"echo '{_END}'\n"
    )


def _build_eval_script(inst: dict, node_ids: list[str]) -> str:
    """Build the eval script for a task, in the SWE-bench apply order.

        1. reset tracked files: git reset --hard <base_commit>
        2. apply the candidate patch
        3. restore the files that test_patch touches (undoing any candidate edits to them),
           remove the files it creates, then apply test_patch last
        4. run the task's test command (install_config.test_cmd) on the FAIL_TO_PASS and
           PASS_TO_PASS node ids
    Because the test patch is applied last, the candidate patch cannot change the graded tests.
    The image has the repository installed at /testbed in a conda `testbed` env. Rows with a
    `verifier_dir` get the Harbor script instead."""
    if inst.get("verifier_dir"):
        return _build_harbor_eval_script(inst)
    base = inst.get("base_commit", "")
    mod_files, new_files = _test_files_from_patch(inst.get("test_patch", ""))
    reset_cmds = []
    if mod_files:                        # files that exist at base: restore them
        reset_cmds.append(f"git checkout {shlex.quote(base)} -- "
                          + " ".join(shlex.quote(f) for f in mod_files))
    if new_files:                        # files the test patch creates: remove candidate copies
        reset_cmds.append("rm -f " + " ".join(shlex.quote(f) for f in new_files))
    reset_tests = "\n".join(reset_cmds) if reset_cmds else "true"
    nodes = " ".join(shlex.quote(n) for n in node_ids)
    test_cmd = _test_cmd(inst)
    return (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source /opt/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate testbed 2>/dev/null || conda activate base\n"
        "cd /testbed\n"
        "git config --global --add safe.directory /testbed\n"
        # 1. reset tracked files to base. The eval container is fresh (the rollout ran in another
        #    container), but a kept container could carry state. No `git clean`: it would delete
        #    untracked files the tests need (conftest, generated package data, a test file that
        #    the test patch adds) and break pytest collection.
        f"git reset --hard {shlex.quote(base)} 2>/dev/null || git reset --hard\n"
        # 2. apply the candidate patch. If it does not apply, the tests fail and the task is
        #    unresolved (not an infrastructure failure); the marker is logged for diagnosis.
        "git apply -v /t2e/model.patch || git apply --3way -v /t2e/model.patch || "
        "patch -p1 --fuzz=5 -i /t2e/model.patch || echo T2E_MODEL_PATCH_FAILED\n"
        # 3. restore the test files, then apply the test patch last so the graded tests are the
        #    task's own.
        f"{reset_tests}\n"
        "git apply -v /t2e/test.patch || git apply --3way -v /t2e/test.patch || echo T2E_TEST_PATCH_FAILED\n"
        f"echo '{_BEGIN}'\n"
        f"{test_cmd} {nodes}\n"
        f"echo '{_END}'\n"
    )


def _run_in_docker(image: str, eval_script: str, model_patch: str, test_patch: str,
                   timeout: int) -> str | None:
    """Run the eval script in a fresh container from `image`, with the patches mounted at /t2e.

    Returns the combined stdout and stderr, or None on an infrastructure failure.
    T2E_SWE_EXEC_BACKEND=chroot uses the dockerless backend in chroot_exec.py instead (same
    contract)."""
    if os.environ.get("T2E_SWE_EXEC_BACKEND", "docker") == "chroot":
        from verse.runtime.chroot_exec import run_in_chroot
        return run_in_chroot(image, eval_script, model_patch, test_patch, timeout)
    docker = os.environ.get("T2E_SWE_DOCKER", "docker")
    # extra `docker run` flags, e.g. "--sysctl net.ipv6.conf.all.disable_ipv6=0" on hosts whose
    # bridge network disables IPv6 loopback in containers (tests that bind ::1 then fail)
    extra_args = os.environ.get("T2E_SWE_DOCKER_RUN_ARGS", "").strip()
    keep = os.environ.get("T2E_SWE_KEEP_CONTAINER", "0") == "1"
    with tempfile.TemporaryDirectory(prefix="t2e_swe_", ignore_cleanup_errors=True) as td:
        with open(os.path.join(td, "model.patch"), "w") as f:
            f.write(model_patch or "")
        with open(os.path.join(td, "test.patch"), "w") as f:
            f.write(test_patch or "")
        with open(os.path.join(td, "eval.sh"), "w") as f:
            f.write(eval_script)
        rm = "" if keep else "--rm"
        # Harbor task: copy tests/ (test.sh, parsers, config.json) into this run's temp dir and
        # mount it at /tests. test.sh writes parser.py there, so each run gets its own copy.
        mounts = ""
        m = re.search("^" + re.escape(_HARBOR_MARK) + r"(.+)$", eval_script, re.M)
        if m:
            vdir = m.group(1).strip()
            if not os.path.isdir(vdir):
                logger.warning(f"[swe] verifier dir missing: {vdir}")
                return None
            shutil.copytree(vdir, os.path.join(td, "tests"))
            mounts = f" -v {shlex.quote(os.path.join(td, 'tests'))}:/tests"
        # --cpus cap: a test suite that runs in parallel (pytest-xdist) can otherwise take most
        # cores and starve the other containers on the host
        cpus = os.environ.get("T2E_SWE_CONTAINER_CPUS", "6")
        # Eval containers run offline by default. Some tasks' test suites need network access;
        # T2E_SWE_JUDGE_ALLOW_NET=1 (or T2E_SWE_ALLOW_NET=1) enables it. The agent never runs in
        # this container, and T2E_SWE_JUDGE_ALLOW_NET does not affect rollout containers.
        net = "" if (os.environ.get("T2E_SWE_ALLOW_NET") == "1"
                     or os.environ.get("T2E_SWE_JUDGE_ALLOW_NET") == "1") else "--network none"
        cmd = (f"{docker} run {rm} {net} {extra_args} --cpus={cpus} -v {shlex.quote(td)}:/t2e{mounts} "
               f"{shlex.quote(image)} /bin/bash /t2e/eval.sh")
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            logger.warning(f"[swe] docker run timed out after {timeout}s for {image}")
            return None
        except Exception as e:
            logger.warning(f"[swe] docker run failed: {e}")
            return None


# pytest -rA short-test-summary lines: "PASSED path::node", "FAILED path::node", etc.
_STATUS_RE = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE)
# Outcome counts in pytest's final summary, e.g. "5 passed", "1 failed", "3 subtests passed".
# Warnings are recognized but excluded from the pass/total counts; _summary_counts merges
# "error" and "errors".
_SUMMARY_RE = re.compile(
    r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|"
    r"subtests?\s+passed|warnings?)", re.IGNORECASE)
# pytest's final summary line, e.g. "=== 5 passed, 1 failed in 3.21s ===", contains " in " and a
# duration. Taking the last such line keeps a stray "N passed" in captured output from being read
# as the summary.
_SUMMARY_LINE_RE = re.compile(r"^=+.*\bin\s+[\d.]+s.*=+\s*$|=+ .*\bin\s+[\d.]+s(?:econds)? =+", re.MULTILINE)


def _parse_pytest_log(inst: dict, log_text: str) -> dict[str, str]:
    """Parse a pytest -rA log into {node_id: STATUS}.

    Uses swebench's per-repo log parser when the repository is registered there; otherwise
    parses the generic -rA short test summary."""
    try:
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        parser = MAP_REPO_TO_PARSER.get(inst.get("repo", ""))
        if parser is not None:
            seg = _between_sentinels(log_text)
            res = parser(seg, inst) if _parser_takes_two(parser) else parser(seg)
            if isinstance(res, dict) and res:
                return res
    except Exception as e:
        logger.debug(f"[swe] swebench parser unavailable/failed, using generic: {e}")
    seg = _between_sentinels(log_text)
    return {node: status for status, node in _STATUS_RE.findall(seg)}


def _parser_takes_two(fn) -> bool:
    import inspect
    try:
        return len(inspect.signature(fn).parameters) >= 2
    except Exception:
        return False


def _between_sentinels(log_text: str) -> str:
    if _BEGIN in log_text and _END in log_text:
        return log_text.split(_BEGIN, 1)[1].split(_END, 1)[0]
    return log_text


def _node_suffix(node_id: str) -> str:
    """Normalize a node id for matching: drop the file path and keep `name[params]`.

    pytest may print `test_x.py::t` while the dataset stores `tests/test_x.py::t` (or the
    reverse); matching on the part after `::` ignores such path differences. Returns the raw id
    when it has no `::`."""
    return node_id.split("::", 1)[1] if "::" in node_id else node_id


def _summary_counts(seg: str) -> dict:
    """Parse pytest's final summary line into {outcome: count}.

    Uses the last line matching '... in <time>s ...' (e.g. '=== 5 passed, 1 failed in 3.21s ==='),
    else the last line with an outcome count and ' in ', else the whole text, and counts the
    outcome keywords on it (passed, failed, error, skipped, xfailed, xpassed, deselected,
    subtests passed, warnings). Callers exclude warnings from the pass/total counts."""
    text = seg or ""
    # the summary line: the last line matching '... in <time>s ...'
    summary_line = None
    for m in _SUMMARY_LINE_RE.finditer(text):
        summary_line = m.group(0)
    if summary_line is None:
        # fallback: the last line with an outcome count and ' in ' (some runners omit the ===)
        for line in reversed(text.splitlines()):
            if " in " in line and _SUMMARY_RE.search(line):
                summary_line = line
                break
    scan = summary_line if summary_line is not None else text
    counts = {}
    for n, kind in _SUMMARY_RE.findall(scan):
        k = kind.lower().strip()
        if k.startswith("subtest"):
            k = "subtests_passed"
        elif k.startswith("warning"):
            k = "warning"
        else:
            k = k.rstrip("s")  # errors->error
        counts[k] = counts.get(k, 0) + int(n)
    return counts


def _resolved_from_log(inst: dict, log_text: str) -> bool:
    """Apply the SWE-bench resolution rule to an eval log.

    Resolved means every FAIL_TO_PASS and PASS_TO_PASS test passes. Two tiers:
      (1) Node-id match: compare the parsed per-test statuses with the expected ids, matching on
          the part after `::` so path differences do not matter. If any expected id appears in
          the log, this tier decides.
      (2) Summary fallback: if no expected id appears (different runner or id format), require
          that the test patch applied and that pytest's final summary shows failed == 0,
          error == 0 and passed >= the number of expected tests. This cannot tell which tests
          passed, but it handles repositories whose printed ids differ from the dataset ids.
    Harbor rows use the reward written by the task's own verifier."""
    if inst.get("verifier_dir"):
        # Harbor task: the verifier already applied the resolution rule and wrote reward.txt,
        # which the eval script prints to the log
        m = re.search(r"T2E_HARBOR_REWARD=(\S+)", log_text or "")
        return bool(m and m.group(1).strip() == "1")
    f2p = set(inst.get("FAIL_TO_PASS", []))
    p2p = set(inst.get("PASS_TO_PASS", []))
    if not f2p:  # no FAIL_TO_PASS tests: the task is ill-defined
        return False

    # A test patch that did not apply makes the run untrustworthy (compute_score treats it as an
    # infrastructure failure). A candidate patch that did not apply needs no check: its tests
    # fail and the task is unresolved. Tier 2 below requires the test patch to have applied,
    # since a summary over a partly applied state could pass falsely.
    test_patch_failed = "T2E_TEST_PATCH_FAILED" in (log_text or "")

    status = _parse_pytest_log(inst, log_text)
    # tier 1: node-id match on the part after `::`. As in SWE-bench's test_passed(), XFAIL counts
    # as passing.
    _PASS_STATES = {"PASSED", "XFAIL"}
    passed_suffix = {_node_suffix(n) for n, s in status.items() if str(s).upper() in _PASS_STATES}
    want = {_node_suffix(t) for t in (f2p | p2p)}
    if want and want.issubset(passed_suffix):
        return True
    # some expected ids appear in the log but not all pass: a real failure
    seen_suffix = {_node_suffix(n) for n in status}
    if want & seen_suffix:
        return False

    # tier 2: no expected id appears (runner or id format mismatch). Used only when the test patch
    # applied; otherwise the counts could come from the tests before the patch and pass falsely.
    if test_patch_failed:
        return False
    counts = _summary_counts(_between_sentinels(log_text))
    if not counts:
        return False
    failed = counts.get("failed", 0) + counts.get("error", 0)
    # passing subtests count as passes; warnings are excluded
    passed = counts.get("passed", 0) + counts.get("subtests_passed", 0)
    n_expected = len(f2p | p2p)
    return failed == 0 and passed >= n_expected and passed > 0


# Scoring entry point.
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Score one rollout by execution.

    Returns 1.0 if the candidate patch resolves the task, 0.0 if it does not (or no patch was
    produced), and INFRA_SENTINEL (NaN) on an infrastructure failure: missing task fields or
    test ids, docker failure or timeout, or a test patch that does not apply. swe_bc_aggregate
    drops NaN from the denominator. data_source and ground_truth are unused; the task comes
    from extra_info.

    T2E_SWE_INFRA_AS_ZERO=1 returns 0.0 for infrastructure failures instead, for callers that
    need a numeric score. T2E_SWE_INFRA_LOG, when set, names a JSONL file that records each
    infrastructure failure with its instance_id and reason."""
    infra_as_zero = os.environ.get("T2E_SWE_INFRA_AS_ZERO", "0") == "1"
    infra_log = os.environ.get("T2E_SWE_INFRA_LOG", "")
    iid = ""
    if isinstance(extra_info, dict):
        iid = str(extra_info.get("instance_id") or extra_info.get("swe_instance", {}).get("instance_id", "")
                  if isinstance(extra_info.get("swe_instance"), dict) else extra_info.get("instance_id", ""))

    def _infra(reason: str) -> float:
        # Record every infrastructure failure (JSONL keyed by instance_id), so these rollouts can
        # be dropped from the denominator later even when a numeric 0.0 is returned.
        logger.warning(f"[swe] INFRA failure ({reason}) inst={iid} -> "
                       f"{'0.0(logged)' if infra_as_zero else 'NaN(drop)'}")
        if infra_log:
            try:
                with open(infra_log, "a") as f:
                    f.write(json.dumps({"instance_id": iid, "reason": reason}) + "\n")
            except Exception:
                pass
        return 0.0 if infra_as_zero else INFRA_SENTINEL

    inst = _as_instance(extra_info)
    if inst is None:
        # missing task fields are a data problem, not an unresolved task
        return _infra("no instance fields in extra_info")
    patch = _extract_patch(extra_info, solution_str)
    if not patch.strip():
        return 0.0  # no patch: unresolved

    timeout = int(os.environ.get("T2E_SWE_RUN_TIMEOUT", "1800"))
    image = _image_name(inst)
    node_ids = list(inst.get("FAIL_TO_PASS", [])) + list(inst.get("PASS_TO_PASS", []))
    if not node_ids:
        return _infra("no FAIL_TO_PASS/PASS_TO_PASS node ids")
    eval_script = _build_eval_script(inst, node_ids)

    log_text = _run_in_docker(image, eval_script, patch, inst.get("test_patch", ""), timeout)
    if log_text is None:
        return _infra("docker run failed/timed out")
    # the test patch did not apply, so the tests cannot be trusted
    if "T2E_TEST_PATCH_FAILED" in log_text:
        return _infra("golden test patch failed to apply")
    return 1.0 if _resolved_from_log(inst, log_text) else 0.0
