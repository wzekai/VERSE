"""CPU unit tests for HookedRuntime, the layer that validates the output of evolved hook code."""
from verse.evolution.hooks_runtime import HookedRuntime

# A hook that appends a text block after the model's tool_use block. The Messages API rejects
# any block after a tool_use in an assistant message, and one such reply breaks every later
# request of the conversation.
_APPEND_AFTER_TOOLUSE = """
class Hooks(BaseHooks):
    def after_llm(self, content, state):
        return list(content) + [{"type": "text", "text": "[harness] note"}]
"""

# Same mistake via in-place mutation: the identity-return fast path must still catch it.
_APPEND_IN_PLACE = """
class Hooks(BaseHooks):
    def after_llm(self, content, state):
        content.append({"type": "text", "text": "[harness] note"})
        return content
"""

_IDENTITY = """
class Hooks(BaseHooks):
    pass
"""


def _reply(turn=0):
    return [{"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": f"toolu_{turn}", "name": "bash",
             "input": {"command": "ls"}}]


def test_after_llm_reorders_text_appended_after_tool_use():
    rt = HookedRuntime(_APPEND_AFTER_TOOLUSE)
    out = rt.after_llm(_reply(), 0)
    types = [b["type"] for b in out]
    # all text first, tool_use last — the API-legal shape
    assert types == ["text", "text", "tool_use"]
    assert out[-1]["id"] == "toolu_0"                      # model's real id preserved
    assert "[harness] note" in out[1]["text"]              # hook's note survives, moved
    assert any(e["hook"] == "after_llm" and "reordered" in e["error"]
               for e in rt.errors)                          # logged, visible to the optimizer


def test_after_llm_reorders_in_place_append_too():
    rt = HookedRuntime(_APPEND_IN_PLACE)
    out = rt.after_llm(_reply(), 0)
    assert [b["type"] for b in out] == ["text", "text", "tool_use"]
    assert any(e["hook"] == "after_llm" for e in rt.errors)


def test_after_llm_identity_and_legal_shapes_untouched():
    rt = HookedRuntime(_IDENTITY)
    content = _reply()
    assert rt.after_llm(content, 0) is content             # identity passes through
    assert rt.errors == []


# Screening tie-breaker
def test_screen_tiebreak_prefers_probe_confirmed_flips():
    from verse.evolution.driver import _probe_confirmed_flips
    from verse.evolution.protocols import Proposal
    from verse.evolution.probes import edits_sha
    from verse.claims import Claim, ExperimentRecord

    edits = [{"path": "workflow.md", "edit": "run tests before submit"}]
    sha = edits_sha(edits)

    # candidate with two probe-confirmed flips on its own final change-set
    verified = Proposal(description="d", meta={"edits": edits})
    verified.claims = [Claim(statement="s", evidence=[ExperimentRecord(
        kind="fix_probe", setup="", outcome="", verdict="supports",
        detail={"edits_sha": sha, "flips": ["t1", "t2"]})])]
    assert _probe_confirmed_flips(verified) == 2

    # baseline candidate without verification records counts 0
    judge = Proposal(description="d", meta={"edits": edits})
    judge.claims = [Claim(statement="s")]
    assert _probe_confirmed_flips(judge) == 0

    # stale evidence: the probe ran on a different (draft) change-set, so it does not count
    stale = Proposal(description="d", meta={"edits": edits})
    stale.claims = [Claim(statement="s", evidence=[ExperimentRecord(
        kind="fix_probe", setup="", outcome="", verdict="supports",
        detail={"edits_sha": "deadbeef00000000", "flips": ["t1"]})])]
    assert _probe_confirmed_flips(stale) == 0

    # exploration-tagged claims (probed-then-discarded drafts) are excluded even on sha match
    explored = Proposal(description="d", meta={"edits": edits})
    explored.claims = [Claim(statement="s", tags={"exploration": True},
                             evidence=[ExperimentRecord(
        kind="fix_probe", setup="", outcome="", verdict="supports",
        detail={"edits_sha": sha, "flips": ["t1"]})])]
    assert _probe_confirmed_flips(explored) == 0

    # the ranking tuple: a tie on net score goes to confirmed flips, not to candidate order
    scores = [(0, _probe_confirmed_flips(judge), -0),      # self_a, no evidence
              (0, _probe_confirmed_flips(verified), -1)]   # self_b, 2 confirmed flips
    assert max(range(2), key=lambda i: scores[i]) == 1
