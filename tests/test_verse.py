"""Tests for VERSE (verification tools plus optimizer self-evolution).

Four groups:
  1. Rendering: contract plus guidance gives the probe prompt of the ours/verified_*
     configurations; plain self_teacher starts from an empty optimizer harness.
  2. Contract-only tools: in VERSE the tool descriptions and the verdict vocabulary carry no
     usage advice; names and schemas match the advice-carrying versions.
  3. seed_wisdom: the round-0 optimizer harness holds exactly the hand-written guidance of the
     ours/verified_* configurations; skills and notes stay empty; the optimizer can edit it.
  4. Shared budget and env passthrough: verification calls and run_episode draw from one
     pool; self-written tools reach the verification tools through env.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verse.evolution.intervener import (
    _PROBE_CONTRACT, _PROBE_CONTRACT_NEUTRAL, _PROBE_TOOLS, _PROBE_TOOLS_CONTRACT,
    PROBE_WISDOM)
from verse.evolution.meta_teacher import (
    MetaTeacher, SharedProbeBudget, TeacherEnv, _seed_equipment)
from verse.evolution.probes import T2TProbeKit


# 1. Rendering
def test_probe_prompt_split_reassembles_verbatim():
    """_PROBE_CONTRACT + PROBE_WISDOM is the probe prompt of the ours/verified_*
    configurations (checked by its first and last lines)."""
    joined = _PROBE_CONTRACT + PROBE_WISDOM
    assert joined.startswith("\n\n## How to use experiment verdicts\n")
    assert "[VERIFIED] = mechanism confirmed" in joined
    assert "fix_probe is your highest-value tool" in joined
    assert joined.rstrip().endswith("propose the full set of edits the evidence supports.")


def test_plain_self_teacher_equipment_unchanged():
    """A self_teacher configuration (no probes, no seed) starts from an empty optimizer
    harness: empty prompt, notes and search files, no skills."""
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True}, d, {"model": "m"}, {"model": "x"})
        for f in ("prompt.md", "notes.md", "search.md"):
            assert mt.ws.read(f) == "", f
        assert mt.ws.skills() == []
        assert mt.seed_wisdom is False


# 2. Contract-only tools
def test_contract_tools_match_names_and_schemas():
    """Same tools and argument schemas; only the descriptions differ."""
    assert [t["name"] for t in _PROBE_TOOLS_CONTRACT] == \
           [t["name"] for t in _PROBE_TOOLS]
    for orig, contract in zip(_PROBE_TOOLS, _PROBE_TOOLS_CONTRACT):
        assert contract["input_schema"] is orig["input_schema"]


def test_contract_tools_carry_no_advice():
    advice_markers = ("USE THIS", "highest-value", "worth far more", "use this to",
                      "your budget", "BEFORE SUBMITTING", "validates your")
    for t in _PROBE_TOOLS_CONTRACT:
        low = t["description"]
        for marker in advice_markers:
            assert marker.lower() not in low.lower(), (t["name"], marker)
    neutral = _PROBE_CONTRACT_NEUTRAL.lower()
    for marker in ("trust and prioritize", "re-diagnose", "verify with a probe"):
        assert marker not in neutral, marker


# 3. seed_wisdom
def test_seed_equipment_is_verified_wisdom_exactly():
    from verse.evolution.driver import _LENSES, _VERIFIED_CODE_RULE
    seed = _seed_equipment()
    assert set(seed) == {"prompt.md", "search.md"}      # nothing beyond the verified_* guidance
    assert seed["prompt.md"] == (PROBE_WISDOM.strip() + "\n\n"
                                 + _VERIFIED_CODE_RULE.strip() + "\n")
    blocks = [b.strip() for b in seed["search.md"].split("---")]
    assert len(blocks) == 3
    for (name, suffix), block in zip(_LENSES, blocks):
        assert block == suffix.strip(), name


def test_verse_init_seeds_and_stays_editable():
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True, "seed_wisdom": True}, d,
                         {"model": "m"}, {"model": "x"})
        assert "fix_probe is your highest-value tool" in mt.ws.read("prompt.md")
        assert mt.ws.read("search.md").count("---") == 2      # 3 blocks
        assert mt.ws.read("notes.md") == ""                    # not seeded
        assert mt.ws.skills() == []
        # search_lenses returns the seeded blocks under self_* names
        lenses = mt.search_lenses(3)
        assert [n for n, _ in lenses] == ["self_a", "self_b", "self_c"]
        assert "OMISSION" in lenses[0][1]
        # the seed is an ordinary optimizer-harness file: an optimizer edit overwrites it
        mt.ws.apply([{"path": "search.md", "content": "my own guidance"}],
                    meta_round=1, description="rewrite")
        assert mt.ws.read("search.md") == "my own guidance"


def test_verse_resume_does_not_reseed():
    """A resumed run (teacher_ws/.git exists) keeps the optimizer's edits: init() does
    not re-apply the seed over them."""
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True, "seed_wisdom": True}, d,
                         {"model": "m"}, {"model": "x"})
        mt.ws.apply([{"path": "prompt.md", "content": "rewritten by teacher"}],
                    meta_round=2, description="edit")
        mt2 = MetaTeacher({"self_edit": True, "seed_wisdom": True}, d,
                          {"model": "m"}, {"model": "x"})
        assert mt2.ws.read("prompt.md") == "rewritten by teacher"


# 4. Shared budget and env passthrough
class _Ctx:
    """Minimal EvolutionContext stand-in for begin_round/TeacherEnv tests."""
    def __init__(self, probes=None):
        self.probes = probes
        self.train_tids = frozenset()
        self.insts_by_tid = {}
        self.scratch_dir = "/tmp"
        self.executor = {}


def test_shared_pool_probes_and_run_episode_draw_together():
    kit = T2TProbeKit(budget=4, budget_s=3600)
    kit.reset_round()
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True, "seed_wisdom": True}, d,
                         {"model": "m"}, {"model": "x"})
        mt.begin_round(1, _Ctx(probes=kit), d, n_candidates=1)
        assert isinstance(mt._budget, SharedProbeBudget)
        assert mt._budget.remaining == 4
        # a verification call spends from the same pool that run_episode sees
        kit._used += 1                       # as a probe call would
        assert mt._budget.remaining == 3
        # and a run_episode slot spends from the probe pool
        assert mt._budget.take() == 2
        assert kit.remaining() == 2


def test_plain_self_teacher_budget_unchanged():
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True, "episode_budget": 8}, d,
                         {"model": "m"}, {"model": "x"})
        mt.begin_round(1, _Ctx(probes=None), d, n_candidates=3)
        assert not isinstance(mt._budget, SharedProbeBudget)
        assert mt._budget.remaining == 24    # 8 per candidate x 3 candidates


def test_env_probe_passthrough():
    """Self-written tools reach the verification tools through env.*, formatted like the
    tool output; without probes (plain self_teacher) these attributes do not exist."""
    class _FakeRec:
        verdict, setup, outcome = "supports", "ran it", "it flipped"

    class _FakeKit:
        def fix_probe(self, ctx, edits, tids): return _FakeRec()
        def ablate(self, ctx, tid, step): return _FakeRec()
        def substitute(self, ctx, tid, step, cmd): return _FakeRec()
        def replay(self, ctx, tid): return _FakeRec()

    env = TeacherEnv(lambda t, edits=None: "ok", lambda c: "ok", lambda p, m=1024: "ok",
                     probes=_FakeKit(), probe_ctx=_Ctx())
    assert env.fix_probe([], ["t1"]) == "[SUPPORTS] ran it -> it flipped"
    assert env.ablate("t1", 3) == "[SUPPORTS] ran it -> it flipped"
    assert env.substitute("t1", 3, "ls") == "[SUPPORTS] ran it -> it flipped"
    assert env.replay("t1") == "[SUPPORTS] ran it -> it flipped"

    bare = TeacherEnv(lambda t, edits=None: "ok", lambda c: "ok", lambda p, m=1024: "ok")
    assert not hasattr(bare, "fix_probe")


def test_verse_equipment_prompt_names_the_shared_pool():
    kit = T2TProbeKit(budget=8, budget_s=3600)
    kit.reset_round()
    with tempfile.TemporaryDirectory() as d:
        mt = MetaTeacher({"self_edit": True, "seed_wisdom": True}, d,
                         {"model": "m"}, {"model": "x"})
        sfx = mt.begin_round(1, _Ctx(probes=kit), d, n_candidates=1)
        assert "ONE shared pool" in sfx
        # plain self_teacher keeps its own budget wording
        mt2 = MetaTeacher({"self_edit": True}, d + "_x" if False else tempfile.mkdtemp(),
                          {"model": "m"}, {"model": "x"})
        sfx2 = mt2.begin_round(1, _Ctx(probes=None), d, n_candidates=1)
        assert "run_episode budget this round" in sfx2
        assert "ONE shared pool" not in sfx2
