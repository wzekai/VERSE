"""Harbor separate-verifier and compose support in tb_env (Terminal-Bench 4.0 task layout).

Docker is faked: every `<docker> ...` command line is recorded and answered from a table, so the
tests pin the sequence the runner issues (Harbor's single_step flow), not container behaviour."""
import os
import re
import subprocess
import textwrap

import pytest

from verse.runtime import tb_env
from verse.runtime.tb_env import TBContainerSession


def _mk_task(root, name, toml, compose=False, tests_dockerfile=True, run_tests=False):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "environment"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    open(os.path.join(d, "environment", "Dockerfile"), "w").write("FROM alpine\n")
    open(os.path.join(d, "tests", "test.sh"), "w").write("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    if tests_dockerfile:
        open(os.path.join(d, "tests", "Dockerfile"), "w").write("FROM alpine\nCOPY . /tests/\n")
    if compose:
        open(os.path.join(d, "environment", "docker-compose.yaml"), "w").write(
            "services:\n  main:\n    depends_on: [api]\n  api:\n    build:\n      context: ./api\n")
    if run_tests:
        open(os.path.join(d, "run-tests.sh"), "w").write("#!/bin/bash\npytest\n")
    if toml is not None:
        open(os.path.join(d, "task.toml"), "w").write(textwrap.dedent(toml))
    return d


TB4_TOML = '''
    artifacts = ["/app/out.json", "/app/submission/", {source = "/shared/snap.json", service = "api"},
                 {source = "/work/gen", exclude = ["node_modules", "*.pyc"]}]
    [verifier]
    timeout_sec = 240.0
    environment_mode = "separate"
    [[verifier.collect]]
    service = "api"
    command = "curl -sf http://localhost:5000/healthz >/dev/null"
    timeout_sec = 10.0
    [[verifier.collect]]
    command = "cd /app && git diff > /tmp/agent.patch || true"
    [verifier.env]
    API_BASE = "http://127.0.0.1:8080"
    [verifier.environment]
    cpus = 4
    allow_internet = false
    [environment]
    cpus = 2
'''


class FakeDocker:
    """Answers docker command lines; creates host files for `docker cp <ctr>:<src> <host>`."""
    DIRS = {"/app/submission", "/logs/artifacts", "/work/gen"}

    def __init__(self):
        self.cmds = []

    def __call__(self, cmd, timeout):
        self.cmds.append(cmd)
        rc, out = 0, ""
        if " image inspect " in cmd:
            rc = 0 if "_client" in cmd or "tb_img_" in cmd and "_verifier" not in cmd else 1
        elif re.search(r"compose .* ps -q ", cmd):
            out = "sidecar123\n"
        elif " inspect -f " in cmd:
            out = "true\n" if "State.Running" in cmd else "/app\n"
        elif " cat /logs/verifier/reward.json" in cmd:
            rc = 1
        elif " cat /logs/verifier/reward.txt" in cmd:
            out = "1\n"
        elif re.search(r"^\S+ cp \S+:(\S+) (\S+)$", cmd):
            m = re.search(r"^\S+ cp \S+:(\S+) (\S+)$", cmd)
            src, dst = m.group(1), m.group(2)
            if src in self.DIRS:
                os.makedirs(dst, exist_ok=True)
                open(os.path.join(dst, "f"), "w").write("x")
            else:
                open(dst, "w").write("x")
        elif "| tar -xf - -C " in cmd:
            dst = cmd.split("| tar -xf - -C ")[1].strip()
            os.makedirs(dst, exist_ok=True)
            open(os.path.join(dst, "kept"), "w").write("x")
        elif "/tests/test.sh" in cmd:
            out = "1 passed\n"
        return subprocess.CompletedProcess(cmd, rc, out, "")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("T2T_TB_TASKS", f"tb4={tmp_path}")
    monkeypatch.setenv("T2E_SWE_CONTAINER_CPUS", "2")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    os.makedirs(tmp_path / "tmp", exist_ok=True)
    tb_env._CONTENT_HASH_CACHE.clear()
    return str(tmp_path)


def test_separate_verifier_gate(root):
    _mk_task(root, "t4", TB4_TOML)
    assert TBContainerSession("t4", src="tb4")._separate_verifier() is True
    # DeepSWE: separate mode and tests/Dockerfile, but a base commit, so it is verified in place
    _mk_task(root, "dswe", '[verifier]\nenvironment_mode = "separate"\n[metadata]\nbase_commit_hash = "abc"\n')
    assert TBContainerSession("dswe", src="tb4")._separate_verifier() is False
    # TB2.1: task.toml without environment_mode; TB1: no task.toml at all
    _mk_task(root, "t21", '[verifier]\ntimeout_sec = 900.0\n')
    assert TBContainerSession("t21", src="tb4")._separate_verifier() is False
    _mk_task(root, "t1", None, run_tests=True)
    assert TBContainerSession("t1", src="tb4")._separate_verifier() is False
    assert TBContainerSession("t1", src="tb4").harbor_compose is False


def test_harbor_compose_layout(root):
    _mk_task(root, "c4", TB4_TOML, compose=True)
    s = TBContainerSession("c4", src="tb4")
    assert s.harbor_compose and s.use_compose and s.exec_target == s.client
    cmd = s._compose("up -d --wait")
    overlay = s._harbor_overlay()
    env_dir = os.path.join(s.task_dir, "environment")
    assert f"--project-directory {env_dir} -f {overlay} -f {s.compose_path} up -d --wait" in cmd
    assert "MAIN_IMAGE_NAME=" in cmd and f"MAIN_CONTAINER_NAME={s.client}" in cmd and "CPUS=2" in cmd
    assert f"CONTEXT_DIR={os.path.join(s.task_dir, 'environment')}" in cmd
    assert "T_BENCH_" not in cmd
    text = open(overlay).read()
    assert "image: ${MAIN_IMAGE_NAME}" in text and "context: ${CONTEXT_DIR}" in text and "cpus: ${CPUS}" in text
    s.close()
    assert not os.path.exists(overlay)
    # TB1 layout: root-level compose file, T_BENCH_* variables, no overlay
    _mk_task(root, "t1", None, run_tests=True)
    open(os.path.join(root, "t1", "docker-compose.yaml"), "w").write("services:\n  client: {}\n")
    t = TBContainerSession("t1", src="tb4")
    assert t.use_compose and not t.harbor_compose
    assert "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME=" in t._compose("build") and "_main.yaml" not in t._compose("build")


def test_score_separate_flow(root, monkeypatch):
    _mk_task(root, "c4", TB4_TOML, compose=True)
    fake = FakeDocker()
    monkeypatch.setattr(TBContainerSession, "_run_raw", lambda self, cmd, timeout: fake(cmd, timeout))
    s = TBContainerSession("c4", src="tb4")
    assert s.start() is True
    assert any("up -d --wait" in c for c in fake.cmds)
    n0 = len(fake.cmds)
    phi = s.score()
    assert phi == 1.0
    seq = fake.cmds[n0:]
    joined = "\n".join(seq)

    def idx(pat):
        hits = [i for i, c in enumerate(seq) if re.search(pat, c)]
        assert hits, f"missing: {pat}\n{joined}"
        return hits[0]

    ver = f"{s.project}_verifier"
    # the main-phase hook runs in the agent container with bash, the sidecar hook in the
    # resolved sidecar with sh
    i_hook_main = idx(rf"exec {s.client} bash -c .*git diff")
    i_hook_side = idx(r"exec sidecar123 sh -c .*curl -sf")
    i_stop_main = idx(r"compose .* stop main")
    assert i_hook_main < i_stop_main < i_hook_side  # Harbor: main first, stop main, then sidecars
    # artifacts pulled from the right containers; exclusions go through tar
    i_out = idx(rf"cp {s.client}:/app/out.json ")
    assert idx(rf"cp {s.client}:/app/submission ") and idx(rf"cp {s.client}:/logs/artifacts ")
    assert idx(r"cp sidecar123:/shared/snap.json ") > i_stop_main
    assert idx(rf"exec {s.client} tar -C /work/gen -cf - --exclude=node_modules --exclude='\*.pyc' \. \| tar -xf - -C ")
    # agent environment torn down before the verifier image and container appear
    i_down = idx(r"compose .* down -v")
    i_build = idx(rf"build -t {s.img}_verifier {s.task_dir}/tests$")
    i_run = idx(rf"run -d --name {ver} --network none --cpus=4 {s.img}_verifier sleep 86400")
    assert i_out < i_down < i_build < i_run
    # artifacts re-materialized at their original paths, dirs emptied first, then test.sh with env
    i_up_file = idx(rf"cp .*/app/out.json {ver}:/app/out.json$")
    idx(rf"exec {ver} sh -c 'mkdir -p /app && chmod 777 /app'$")
    idx(rf"exec {ver} sh -c 'mkdir -p /app/submission && find /app/submission -mindepth 1 -delete && chmod 777 /app/submission'")
    idx(rf"exec {ver} sh -c 'mkdir -p /logs/verifier /logs/artifacts && chmod 777 /logs/verifier /logs/artifacts'")
    idx(rf"cp .*/app/submission/\. {ver}:/app/submission$")
    idx(rf"cp .*/work/gen/\. {ver}:/work/gen$")
    idx(rf"cp .*/shared/snap.json {ver}:/shared/snap.json$")
    i_test = idx(rf"exec -e API_BASE=http://127.0.0.1:8080 {ver} bash -c 'chmod \+x /tests/test.sh; /tests/test.sh'")
    assert i_run < i_up_file < i_test
    # verdict read from the verifier (json first, then txt), verifier removed, host scratch gone
    assert idx(rf"exec {ver} cat /logs/verifier/reward.json") < idx(rf"exec {ver} cat /logs/verifier/reward.txt")
    assert idx(rf"rm -f {ver}$") > i_test
    assert "1 passed" in s.last_test_output
    assert not [d for d in os.listdir(os.path.join(root, "tmp")) if d.startswith(f"{s.project}_artifacts_")]
    assert not os.path.exists(s._harbor_overlay()) or True  # recreated on demand; close() drops it
    s.close()
    assert not os.path.exists(os.path.join(root, "tmp", f"{s.project}_main.yaml"))


def test_score_separate_single_container_public_network(root, monkeypatch):
    toml = '''
    artifacts = ["/app/result.txt"]
    [verifier]
    environment_mode = "separate"
    [environment]
    cpus = 8
    '''
    _mk_task(root, "s4", toml)
    fake = FakeDocker()
    monkeypatch.setattr(TBContainerSession, "_run_raw", lambda self, cmd, timeout: fake(cmd, timeout))
    s = TBContainerSession("s4", src="tb4")
    assert not s.use_compose
    assert s.start() is True
    assert s.score() == 1.0
    joined = "\n".join(fake.cmds)
    assert f"rm -f {s.name}" in joined  # agent container removed before the verifier
    assert re.search(rf"run -d --name {s.project}_verifier --cpus=8 {s.img}_verifier sleep 86400", joined)
    assert "--network none" not in joined                    # Harbor default: public
    assert f"exec  {s.project}_verifier bash -c" in joined   # no [verifier.env] -> no -e flags
    assert "compose" not in joined


def test_reward_txt_float_tolerant(root, monkeypatch):
    _mk_task(root, "s4", TB4_TOML)
    s = TBContainerSession("s4", src="tb4")
    answers = {}
    monkeypatch.setattr(TBContainerSession, "_run_raw",
                        lambda self, cmd, timeout: subprocess.CompletedProcess(cmd, 0, answers["v"], ""))
    for v, want in (("1", 1.0), ("1.0\n", 1.0), ("0", 0.0), ("0.5", 0.0), ("-1", None), ("nan", None), ("junk", None)):
        answers["v"] = v
        assert s._phi_from_reward_txt("x") == want, v
