"""CPU unit tests for verse.workspace: the executor harness, edit manifests, verdicts, revert."""
import os
import subprocess
import tempfile

import pytest

from verse.workspace import HarnessWorkspace, Manifest, verdict


def _ws():
    d = tempfile.mkdtemp(prefix="hws_")
    ws = HarnessWorkspace(d)
    ws.init_blank(base_prompt="You are a software engineer agent.")
    return ws


def test_init_and_assemble_roundtrip():
    ws = _ws()
    text = ws.assemble()
    assert "software engineer" in text
    # empty components are skipped, not rendered as empty headers
    assert "## Memory" not in text


def test_apply_manifest_commits_and_assembles():
    ws = _ws()
    m = Manifest(component="skills", path="skills/search-first.md",
                 edit="Always grep the repo before editing.",
                 predicted_fixes=["t1", "t2"], at_risk=["t9"])
    sha = ws.apply_manifest(m, round_idx=1)
    assert len(sha) == 40
    assert "search-first" in ws.assemble()
    # manifest json is recoverable from the commit body
    body = subprocess.run(["git", "-C", ws.root, "log", "-1", "--format=%B"],
                          capture_output=True, text=True).stdout
    assert "predicted_fixes" in body and "t1" in body


def test_revert_last_restores_content():
    ws = _ws()
    ws.apply_manifest(Manifest(component="memory.md", path="memory.md", edit="bad memory"), 1)
    assert "bad memory" in ws.assemble()
    ws.revert_last()
    assert "bad memory" not in ws.assemble()


def test_manifest_rejects_path_escape_and_unknown_component():
    with pytest.raises(ValueError):
        Manifest(component="skills", path="../../etc/passwd", edit="x")
    with pytest.raises(ValueError):
        Manifest(component="rootkit", path="rootkit", edit="x")


def test_verdict_labels():
    m = Manifest(component="skills", path="skills/s.md", edit="x",
                 predicted_fixes=["a"], at_risk=[])
    # EFFECTIVE: predicted fix hit, no regression
    v = verdict(m, before={"a": 0, "b": 1}, after={"a": 1, "b": 1})
    assert v["label"] == "EFFECTIVE" and v["predicted_fixes_hit"] == ["a"] and v["delta_phi"] > 0
    # HARMFUL: unpredicted regression
    v = verdict(m, before={"a": 0, "b": 1}, after={"a": 0, "b": 0})
    assert v["label"] == "HARMFUL" and v["unpredicted_regressions"] == ["b"]
    # INERT
    v = verdict(m, before={"a": 0}, after={"a": 0})
    assert v["label"] == "INERT"
    # MIXED
    v = verdict(m, before={"a": 0, "b": 1}, after={"a": 1, "b": 0})
    assert v["label"] == "MIXED"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
