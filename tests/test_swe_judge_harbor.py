"""swe_judge and swe_env on Harbor-packaged SWE-rebench V2 tasks, graded by each task's verifier."""
import os, re, subprocess, types
import pytest

from verse.runtime import swe_judge as J


def _inst(tmp_path):
    vdir = tmp_path / "tests"; vdir.mkdir()
    (vdir / "test.sh").write_text("#!/bin/bash\necho verifier\n")
    (vdir / "config.json").write_text("{}")
    return {"instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": "abc123",
            "test_patch": "diff --git a/tests/t_new.py b/tests/t_new.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/t_new.py\n@@ -0,0 +1 @@\n+x\n",
            "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": [], "repo_dir": "/repo", "verifier_dir": str(vdir),
            "docker_image": "sweb07/owner__repo-1:v0.1.0", "image_name": "docker.io/swerebenchv2/owner__repo-1:v0.1.0"}


def test_harbor_eval_script_dispatch_and_contents(tmp_path):
    inst = _inst(tmp_path)
    s = J._build_eval_script(inst, ["t1"])
    assert s.splitlines()[1] == J._HARBOR_MARK + inst["verifier_dir"]
    assert "ln -sfn /repo /testbed" in s and "\ncd /repo\n" in s
    assert "git reset --hard abc123" in s and "/t2e/model.patch" in s  # canonical order kept
    assert "rm -f tests/t_new.py" in s  # golden tests reset before test.sh
    assert "bash /tests/test.sh" in s and "T2E_HARBOR_REWARD=" in s
    assert s.index("/t2e/model.patch") < s.index("bash /tests/test.sh")
    assert J._BEGIN in s and J._END in s
    # a SWE-rebench V1 instance (no verifier_dir) keeps the generic recipe byte-for-byte
    v1 = {k: v for k, v in inst.items() if k not in ("verifier_dir", "repo_dir")}
    g = J._build_eval_script(v1, ["t1"])
    assert J._HARBOR_MARK not in g and "cd /testbed" in g and "python -m pytest" in g


def test_harbor_resolved_from_reward(tmp_path):
    inst = _inst(tmp_path)
    assert J._resolved_from_log(inst, f"{J._BEGIN}\nT2E_HARBOR_EXIT=0\nT2E_HARBOR_REWARD=1\n{J._END}") is True
    assert J._resolved_from_log(inst, "T2E_HARBOR_REWARD=0") is False
    assert J._resolved_from_log(inst, "T2E_HARBOR_REWARD=none\nT2E_TEST_PATCH_FAILED") is False
    # V1 instance: pytest -rA parsing decides
    v1 = {k: v for k, v in inst.items() if k not in ("verifier_dir", "repo_dir")}
    v1["FAIL_TO_PASS"] = ["tests/t.py::test_a"]
    assert J._resolved_from_log(v1, f"{J._BEGIN}\nPASSED tests/t.py::test_a\n{J._END}") is True


def test_run_in_docker_mounts_verifier_copy(tmp_path, monkeypatch):
    inst = _inst(tmp_path)
    script = J._build_eval_script(inst, ["t1"])
    seen = {}

    def fake_run(cmd, shell, capture_output, text, timeout):
        seen["cmd"] = cmd
        m = re.search(r"-v (\S+):/tests", cmd); assert m, cmd
        assert os.path.isfile(os.path.join(m.group(1), "test.sh"))          # per-run copy of tests/
        assert os.path.realpath(m.group(1)) != os.path.realpath(inst["verifier_dir"])
        return types.SimpleNamespace(stdout="T2E_HARBOR_REWARD=1", stderr="")
    monkeypatch.setattr(J.subprocess, "run", fake_run)
    monkeypatch.setenv("T2E_SWE_EXEC_BACKEND", "docker")
    out = J._run_in_docker(inst["docker_image"], script, "diff", inst["test_patch"], 60)
    assert "T2E_HARBOR_REWARD=1" in out and "--network none" in seen["cmd"] and ":/t2e" in seen["cmd"]
    # generic script: no /tests mount
    v1 = {k: v for k, v in inst.items() if k not in ("verifier_dir", "repo_dir")}
    def fake_run2(cmd, shell, capture_output, text, timeout):
        assert ":/tests" not in cmd
        return types.SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(J.subprocess, "run", fake_run2)
    J._run_in_docker("img", J._build_eval_script(v1, ["t1"]), "diff", "", 60)


def test_image_name_prefers_local_built_image(monkeypatch, tmp_path):
    inst = _inst(tmp_path)
    monkeypatch.delenv("T2E_SWE_ECR_PREFIX", raising=False)
    assert J._image_name(inst) == "sweb07/owner__repo-1:v0.1.0"


def test_swe_env_start_aliases_testbed(monkeypatch):
    from verse.runtime import swe_env as E
    calls = []
    sess = E.SWEContainerSession("sweb07/x:v0.1.0", name="t2e_test")
    def fake_run(args, timeout=60):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sess, "_run", fake_run)
    monkeypatch.setenv("T2E_SWE_SCRUB_GIT", "0")
    assert sess.start() is True
    alias = [c for c in calls if "ln -sfn" in c]
    assert len(alias) == 1 and "/testbed/.git" in alias[0] and "ls -d /*/.git" in alias[0]
    assert calls.index(alias[0]) < [i for i, c in enumerate(calls) if "safe.directory" in c][0]


def test_harbor_eval_script_exports_verifier_env(tmp_path):
    """task.toml [verifier.env] (Java tasks: _JAVA_OPTIONS turning IPv6 preference off) is exported
    right before the verifier runs; without a [verifier.env] the script is byte-identical."""
    task = tmp_path / "floci-io__floci-1021"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")
    inst = {"instance_id": "floci-io__floci-1021", "base_commit": "abc", "repo": "floci-io/floci",
            "repo_dir": "/floci", "verifier_dir": str(task / "tests"), "test_patch": ""}
    plain = J._build_eval_script(inst, [])
    assert "export " not in plain
    (task / "task.toml").write_text(
        '[verifier]\ntimeout_sec = 3000.0\n\n[verifier.env]\n'
        '_JAVA_OPTIONS = "-Dfile.encoding=UTF-8 -Djava.net.preferIPv6Addresses=false"\n'
        '"bad key" = "ignored"\n\n[environment]\nnetwork_mode = "public"\n')
    assert J._harbor_verifier_env(inst) == {"_JAVA_OPTIONS": "-Dfile.encoding=UTF-8 -Djava.net.preferIPv6Addresses=false"}
    s = J._build_eval_script(inst, [])
    exp = "export _JAVA_OPTIONS='-Dfile.encoding=UTF-8 -Djava.net.preferIPv6Addresses=false'\n"
    assert exp in s and "bad key" not in s
    assert s.index(exp) < s.index("bash /tests/test.sh")
    assert s.replace(exp, "") == plain
    (task / "task.toml").write_text('[verifier]\ntimeout_sec = 3000.0\n\n[verifier.env]\n')
    assert J._build_eval_script(inst, []) == plain
    (task / "task.toml").write_text('not = valid = toml\n')
    assert J._build_eval_script(inst, []) == plain
