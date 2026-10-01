"""Component protocols and the shared context and result types.

Plain dataclasses and duck-typed protocols: a component is any object with the right methods.
The driver builds components from the yaml config via the registry. This module imports no
models or containers; the components carry their own dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Shared context
@dataclass
class EvolutionContext:
    """Everything a component may need about the current round, assembled by the driver.

    The context holds paths rather than loaded data: pull-style evidence sources (pull_explore)
    hand paths to a coding agent, and push-style sources load what they need. All four baseline
    implementations likewise keep trajectories as files on disk.
    """
    round_idx: int
    ws_dir: str                                  # the executor harness workspace (git repo)
    traj_dir: str                                # this round's trajectory dumps (one json per task)
    train_outcomes: dict = field(default_factory=dict)     # {task_id: 0.0|1.0} current train sweep
    prev_train_outcomes: dict = field(default_factory=dict)
    insts_by_tid: dict = field(default_factory=dict)       # {task_id: swe instance dict}
    train_tids: frozenset = frozenset()          # ids the tools may run on (train split only).
    # Val ids are also in insts_by_tid for the gate, but val selects the final round, so the
    # optimizer must not run tools on val tasks.
    fail_transcripts: dict = field(default_factory=dict)   # {task_id: transcript list}
    success_transcripts: dict = field(default_factory=dict)  # {task_id: transcript} passing tasks
    # every configuration can read these; running tools on them (minimize, fix_probe) needs
    # the verification tools
    history: list = field(default_factory=list)            # prior round records (dicts)
    probes: object = None                        # ProbeKit if the config includes one, else None
    scratch_dir: str = ""                        # per-round scratch for reports/artifacts
    executor: dict = field(default_factory=dict) # executor config {model, regions, max_turns}
    extra: dict = field(default_factory=dict)    # gate splits, run labels, other driver data


@dataclass
class Proposal:
    """One proposed harness change set: a workspace commit plus its predict-then-verify record.

    Covers AHE's change manifest, Meta-Harness's pending_eval candidate and Self-Harness's
    proposal bundle: the fields are those all three carry, plus the prediction fields.
    """
    description: str
    rationale: str = ""                          # root-cause reasoning (claims text)
    files: list = field(default_factory=list)    # workspace-relative files touched
    predicted_fixes: list = field(default_factory=list)
    at_risk: list = field(default_factory=list)
    claims: list = field(default_factory=list)   # the Claim objects behind the rationale
    raw: str = ""                                # the optimizer's full final message (audit)
    meta: dict = field(default_factory=dict)     # source-specific extras (hook family, ...)


@dataclass
class GateVerdict:
    keep: bool
    label: str                                   # EFFECTIVE/MIXED/INERT/HARMFUL/ACCEPTED/...
    reason: str = ""
    detail: dict = field(default_factory=dict)   # per-split deltas, flip lists, scores


# Slot protocols
class EvidenceSource:
    """Evidence slot: how the optimizer learns about this round's failures.

    gather(ctx) -> (report_text, claims). report_text goes into the optimizer's prompt; claims
    are the structured records behind it, saved for auditing. Implementations include
    push_tails, push_digest_layered (AHE), push_clustered (Self-Harness) and pull_explore
    (Meta-Harness; returns instructions and file access instead of a pre-built report)."""

    def gather(self, ctx: EvolutionContext) -> tuple:  # (str, list[Claim])
        raise NotImplementedError


class EditSpace:
    """Edit-space slot: what the optimizer may change and how an edit is applied and reverted.

    apply(proposal_payload, ws_dir, round_idx) -> commit sha (raises on invalid edits);
    revert_last(ws_dir); assemble(ws_dir) -> the system prompt the executor sees."""

    def apply(self, payload: dict, ws_dir: str, round_idx: int) -> str:
        raise NotImplementedError

    def revert_last(self, ws_dir: str) -> None:
        raise NotImplementedError

    def assemble(self, ws_dir: str) -> str:
        raise NotImplementedError


class Gate:
    """Gate slot: keep or revert. judge(ctx, proposal, before, after) -> GateVerdict.

    before/after are {task_id: phi} on the split the gate's config declares; the driver runs
    the sweeps and the gate only decides. Implementations: keep_all (the default),
    val_net_improve, manifest_falsify (AHE), dual_split_noregress (Self-Harness) and
    tolerance_gate (HarnessX)."""

    def judge(self, ctx: EvolutionContext, proposal: Proposal,
              before: dict, after: dict) -> GateVerdict:
        raise NotImplementedError
