"""Optimizer-episode wiring of the tool-scheduling switches, with a mocked LLM.

With a scheduler in ctx.extra the verification tools are not offered to the model, the
scheduled results are in the prompt, and the submitted draft gets a policy-chosen fix_probe
whose record lands in the proposal's final evidence. With a tool subset only the configured
tools are offered.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import verse.evolution.intervener as iv  # noqa: E402
from verse.evolution.protocols import EvolutionContext
from verse.evolution.probe_scheduler import FixedProbeScheduler
from test_probe_agency_switches import _StubKit, _ctx


def _mock_bedrock(captured):
    """Immediately submit a one-file proposal; capture every request body."""

    def fake(model, regions, body):
        captured.append(body)
        return {"content": [{"type": "tool_use", "id": "t1", "name": "submit_proposal",
                             "input": {"description": "d", "rationale": "r",
                                       "edits": [{"path": "system_prompt.md",
                                                  "content": "x"}],
                                       "predicted_fixes": ["t3"]}}]}

    return fake


def _run(ctx, monkey_target, captured):
    import verse.runtime.executor as rx
    real = rx._invoke_bedrock_raw
    rx._invoke_bedrock_raw = monkey_target
    try:
        with tempfile.TemporaryDirectory() as d:
            return iv.run_intervener("investigate.", d, ctx,
                                     model="stub", regions="us-west-2",
                                     max_turns=3, hooks_pipeline=False)
    finally:
        rx._invoke_bedrock_raw = real


def test_scheduler_episode_unmounts_tools_and_probes_the_draft():
    kit = _StubKit(budget=8)
    ctx = _ctx(kit)
    ctx.extra["probe_scheduler"] = FixedProbeScheduler()
    captured = []
    prop = _run(ctx, _mock_bedrock(captured), captured)
    tool_names = {t["name"] for t in captured[0]["tools"]}
    assert not tool_names & {"replay", "ablate", "substitute", "fix_probe"}
    prompt = captured[0]["messages"][0]["content"]
    assert "Scheduled experiment results" in prompt
    kinds = [k for k, _, _ in kit.calls]
    assert kinds[:2] == ["replay", "replay"] and kinds[-1] == "fix_probe"
    assert prop.meta["n_experiments"] == len(kit.calls)  # on-submit probe included
    fix_recs = [r for c in prop.claims for r in c.evidence if r.kind == "fix_probe"]
    assert fix_recs, "the on-submit fix_probe must land in the final evidence"


def test_subset_episode_mounts_only_configured_tools():
    kit = _StubKit(budget=8)
    kit.tools = ("fix_probe",)
    ctx = _ctx(kit)
    captured = []
    _run(ctx, _mock_bedrock(captured), captured)
    tool_names = {t["name"] for t in captured[0]["tools"]}
    assert "fix_probe" in tool_names
    assert not tool_names & {"replay", "ablate", "substitute"}
    prompt = captured[0]["messages"][0]["content"]
    assert "highest-value tool" in prompt              # fix_probe guidance kept
    assert "ablate PASSING" not in prompt              # ablate guidance dropped
