"""Config switches of the tool-scheduling and self-evolution ablations (no LLM, no docker).

probes.tools loads a subset of the verification tools; probe_scheduler replaces the
optimizer's choice of verification runs with a random or fixed policy;
meta_teacher.self_edit_channels restricts which parts of the optimizer harness self-evolution
may write.
"""
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verse.evolution import build_component, registered
from verse.evolution.intervener import PROBE_WISDOM, _wisdom_for
from verse.evolution.probes import PROBE_TOOL_NAMES, T2TProbeKit
from verse.evolution.probe_scheduler import (
    FixedProbeScheduler, RandomProbeScheduler, ON_SUBMIT_RESERVE,
)
from verse.evolution.protocols import EvolutionContext
from verse.evolution.meta_teacher import TeacherWorkspace, _CHANNEL_PATHS

CFG = os.path.join(os.path.dirname(__file__), "..", "verse", "configs")


def _load(*rel):
    return yaml.safe_load(open(os.path.join(CFG, *rel)))


# probes.tools subset

def test_probekit_default_mounts_all_four():
    assert T2TProbeKit().tools == PROBE_TOOL_NAMES


def test_probekit_subset_validated():
    assert T2TProbeKit(tools=["fix_probe"]).tools == ("fix_probe",)
    try:
        T2TProbeKit(tools=["fix_probe", "teleport"])
    except ValueError as e:
        assert "teleport" in str(e)
    else:
        raise AssertionError("unknown tool name accepted")


def test_wisdom_full_kit_is_byte_identical():
    assert _wisdom_for(PROBE_TOOL_NAMES) is PROBE_WISDOM
    assert _wisdom_for(None) is PROBE_WISDOM


def test_wisdom_subset_drops_and_renumbers():
    w = _wisdom_for(("fix_probe",))
    assert "fix_probe is your highest-value tool" in w
    assert "ablate PASSING" not in w                  # ablate item dropped
    assert "Attribute layer" not in w                 # needs substitute too
    assert "2. Do NOT spend budget" in w              # budget item renumbered 4 -> 2
    w2 = _wisdom_for(("ablate", "replay", "substitute"))
    assert "fix_probe is your highest-value tool" not in w2
    assert "1. ablate PASSING" in w2


# Probe schedulers

class _StubKit:
    """Stand-in for T2TProbeKit: same call surface; records each call and spends one unit."""

    def __init__(self, budget=8):
        self.budget = budget
        self._used = 0
        self.calls = []

    def remaining(self):
        return max(0, self.budget - self._used)

    def _rec(self, kind, task_id, **detail):
        self._used += 1
        self.calls.append((kind, task_id, detail))

        class R:
            pass

        r = R()
        r.kind, r.task_id = kind, task_id
        r.verdict, r.setup, r.outcome, r.detail = "inconclusive", kind, "stub", detail
        return r

    def replay(self, ctx, tid):
        return self._rec("replay", tid)

    def ablate(self, ctx, tid, step):
        return self._rec("ablate", tid, step=step)

    def fix_probe(self, ctx, edits, targets):
        return self._rec("fix_probe", ",".join(targets), n_edits=len(edits))


def _ctx(kit, failed=("t1", "t2", "t3", "t4"), steps=5):
    ctx = EvolutionContext(round_idx=1, ws_dir="", traj_dir="")
    ctx.probes = kit
    ctx.train_outcomes = {t: 0.0 for t in failed}
    ctx.train_outcomes["p1"] = 1.0
    ctx.train_tids = frozenset(list(failed) + ["p1"])
    ctx.fail_transcripts = {t: [{"role": "assistant"}] * steps for t in failed}
    return ctx


def test_scheduler_kinds_registered():
    assert registered("probe_scheduler") == [("probe_scheduler", "fixed"),
                                             ("probe_scheduler", "random")] or \
        set(registered("probe_scheduler")) >= {"fixed", "random"}


def test_random_scheduler_respects_submit_reserve_and_seed():
    kit = _StubKit(budget=8)
    recs = RandomProbeScheduler(seed=0).pre_episode(_ctx(kit))
    assert len(recs) == 8 - ON_SUBMIT_RESERVE
    assert kit.remaining() == ON_SUBMIT_RESERVE
    kit2 = _StubKit(budget=8)
    recs2 = RandomProbeScheduler(seed=0).pre_episode(_ctx(kit2))
    assert [(r.kind, r.task_id) for r in recs] == \
           [(r.kind, r.task_id) for r in recs2]      # same seed and round, same draws


def test_random_on_submit_runs_one_fix_probe():
    kit = _StubKit(budget=8)
    out = RandomProbeScheduler(seed=0).on_submit(
        _ctx(kit), [{"path": "a", "content": "x"}], [])
    assert [r.kind for r in out] == ["fix_probe"]
    assert RandomProbeScheduler(seed=0).on_submit(_ctx(_StubKit()), [], []) == []


def test_fixed_scheduler_is_deterministic_policy():
    kit = _StubKit(budget=8)
    recs = FixedProbeScheduler().pre_episode(_ctx(kit))
    assert [(r.kind, r.task_id) for r in recs[:2]] == \
           [("replay", "t1"), ("replay", "t2")]      # sorted failures, replay first
    assert all(r.kind == "ablate" and r.detail["step"] == 4 for r in recs[2:])
    out = FixedProbeScheduler().on_submit(
        _ctx(_StubKit()), [{"path": "a", "content": "x"}], ["t3", "p1", "t9"])
    assert out[0].task_id == "t3"                    # a predicted fix that failed wins


def test_scheduler_configs_differ_from_base_only_in_scheduler():
    base = _load("ours", "verified_ahe.yaml")
    for name, kind in (("verified_ahe_randprobe.yaml", "random"),
                       ("verified_ahe_fixedsched.yaml", "fixed")):
        cfg = _load("ours", name)
        assert cfg.pop("probe_scheduler")["kind"] == kind
        assert cfg["probes"] == base["probes"]
        assert cfg["evidence"] == base["evidence"]
        assert cfg["attribute"] == base["attribute"]
        assert build_component("probe_scheduler",
                               _load("ours", name)["probe_scheduler"]) is not None


# Self-edit channels

def test_channel_union_and_rejection():
    d = tempfile.mkdtemp()
    ws = TeacherWorkspace(d, channels=["prompt"])
    ws.init({"t": "x"})
    assert ws.editable == {"prompt.md", "search.md"}
    assert "outside this arm's self-edit channels" in ws.validate_edits(
        [{"path": "notes.md", "content": "x"}], [])
    assert ws.validate_edits([{"path": "prompt.md", "content": "x"}], []) == ""
    assert TeacherWorkspace(d).editable >= set().union(*_CHANNEL_PATHS.values())


def test_channel_configs_differ_from_base_only_in_channels():
    base = _load("self_teacher", "self_teacher.yaml")["meta_teacher"]
    for name, ch in (("self_teacher_promptonly.yaml", ["prompt"]),
                     ("self_teacher_toolsonly.yaml", ["tools"]),
                     ("self_teacher_notesonly.yaml", ["notes"])):
        mt = _load("self_teacher", name)["meta_teacher"]
        assert mt.pop("self_edit_channels") == ch
        assert mt == base
