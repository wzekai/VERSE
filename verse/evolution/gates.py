"""gates.py — the keep/revert gates applied to each round's edit.

  keep_all              keep every valid edit and record its label; the best round is picked
                        at the end (Meta-Harness frontier selection). Used by every config
                        through _base.yaml.
  val_net_improve       keep iff more val tasks are fixed than regressed
  dual_split_noregress  Self-Harness (acceptance/run_acceptance_gate.py:88-136): no split's
                        mean pass rate drops and at least one improves
  tolerance_gate        HarnessX (gaia_evolver/run.py:1174-1257): keep iff
                        score >= best - tolerance, with a pass-count noise guard; the best
                        advances only on strict improvement
  manifest_falsify      AHE (evolve.py:2283-2303): attribution verdict per change. In AHE the
                        rollback is advisory and left to the agent in the next round; here
                        HARMFUL edits are reverted mechanically to keep a single lineage.

All gates are pure decision logic over {task_id: phi} dicts; the driver runs all sweeps.
"""
from __future__ import annotations

from verse.evolution.protocols import Gate, GateVerdict
from verse.evolution.registry import register_component


def _flips(before: dict, after: dict) -> tuple:
    common = set(before) & set(after)
    fixed = sorted(t for t in common if before[t] < 0.5 <= after[t])
    regressed = sorted(t for t in common if after[t] < 0.5 <= before[t])
    return fixed, regressed


@register_component("gate", "val_net_improve")
class ValNetImproveGate(Gate):
    """Keep iff strictly more tasks are fixed than regressed on the (repo-disjoint) val
    sweep. Flip confirmation is done by the driver before it calls the gate."""

    def judge(self, ctx, proposal, before, after) -> GateVerdict:
        fixed, regressed = _flips(before, after)
        keep = len(fixed) > len(regressed)
        label = ("EFFECTIVE" if fixed and not regressed else
                 "HARMFUL" if regressed and not fixed else
                 "MIXED" if fixed and regressed else "INERT")
        return GateVerdict(keep=keep, label=label,
                           reason=f"net {len(fixed)}-{len(regressed)} on val",
                           detail={"fixed": fixed, "regressed": regressed})


@register_component("gate", "keep_all")
class KeepAllGate(Gate):
    """Keep every valid proposal (Meta-Harness frontier selection: no candidate is discarded,
    and the best-val round is picked from the journal at the end). Labels are still computed
    and journaled for the final selection and later analysis. With this gate, an edit based
    on a wrong diagnosis stays in the lineage, so checks before submission matter most."""

    def judge(self, ctx, proposal, before, after) -> GateVerdict:
        fixed, regressed = _flips(before, after)
        label = ("EFFECTIVE" if fixed and not regressed else
                 "HARMFUL" if regressed and not fixed else
                 "MIXED" if fixed and regressed else "INERT")
        return GateVerdict(keep=True, label=label,
                           reason=f"keep_all: net {len(fixed)}-{len(regressed)} recorded, not gated",
                           detail={"fixed": fixed, "regressed": regressed})


@register_component("gate", "dual_split_noregress")
class DualSplitNoRegressGate(Gate):
    """Self-Harness acceptance rule: accept iff no split's mean pass rate drops and at least
    one split improves. Split membership is read from ctx.extra[splits_key]
    ({split_name: set(tids)}); without it the gate uses a single split over all tasks."""

    def __init__(self, splits_key: str = "gate_splits"):
        self.splits_key = splits_key

    def judge(self, ctx, proposal, before, after) -> GateVerdict:
        splits = (ctx.extra or {}).get(self.splits_key)
        if not splits:
            # default: a single split over all tasks (reduces to delta > 0)
            splits = {"train": set(before)}
        statuses = {}
        for name, tids in splits.items():
            b = [before[t] for t in tids if t in before]
            a = [after[t] for t in tids if t in after]
            if not b or not a:
                statuses[name] = ("unchanged", 0.0)
                continue
            delta = sum(a) / len(a) - sum(b) / len(b)
            statuses[name] = ("improved" if delta > 0 else "dropped" if delta < 0 else "unchanged",
                              delta)
        dropped = [n for n, (s, _) in statuses.items() if s == "dropped"]
        improved = [n for n, (s, _) in statuses.items() if s == "improved"]
        keep = not dropped and bool(improved)
        reason = (f"accepted: improved {improved} with no split drops" if keep else
                  f"rejected: dropped {dropped}" if dropped else "rejected: no split improved")
        return GateVerdict(keep=keep, label="ACCEPTED" if keep else "REJECTED", reason=reason,
                           detail={n: {"status": s, "delta": d} for n, (s, d) in statuses.items()})


@register_component("gate", "tolerance_gate")
class ToleranceGate(Gate):
    """HarnessX gate (gaia variant): keep iff score >= best_score - tolerance, with a
    pass-count noise guard; the best advances only on strict improvement. On revert,
    detail['revert_to_round'] names the historical best round. (The driver itself resets
    to the pre-round state and does not read this field.)"""

    def __init__(self, tolerance: float = 0.03, noise_threshold: int = 3):
        self.tolerance = float(tolerance)
        self.noise_threshold = int(noise_threshold)
        self._best = None       # (score, passed, round_idx)

    def judge(self, ctx, proposal, before, after) -> GateVerdict:
        passed = sum(1 for v in after.values() if v >= 0.5)
        score = passed / max(1, len(after))
        if self._best is None and ctx.history:
            # resume: _best lives only in memory, so rebuild it from journaled verdict scores
            seen = [(h["verdict"]["score"], h["round"]) for h in ctx.history
                    if isinstance(h.get("verdict"), dict) and "score" in h["verdict"]]
            if seen:
                b_score, b_round = max(seen)
                self._best = (b_score, round(b_score * max(1, len(after))), b_round)
        if self._best is None:
            self._best = (score, passed, ctx.round_idx)
            return GateVerdict(True, "ACCEPTED", "first round — establishing baseline",
                               {"score": score})
        best_score, best_passed, best_round = self._best
        if score < best_score - self.tolerance:
            if abs(passed - best_passed) < self.noise_threshold:
                return GateVerdict(True, "ACCEPTED",
                                   f"noise-level regression (|Δpassed|={abs(passed-best_passed)} "
                                   f"< {self.noise_threshold})", {"score": score})
            return GateVerdict(False, "REVERTED",
                               f"score {score:.3f} < best {best_score:.3f} - tol {self.tolerance}",
                               {"score": score, "revert_to_round": best_round})
        if score > best_score:
            self._best = (score, passed, ctx.round_idx)
        return GateVerdict(True, "ACCEPTED",
                           f"score {score:.3f} >= best {best_score:.3f} - tol {self.tolerance}",
                           {"score": score})


@register_component("gate", "manifest_falsify")
class ManifestFalsifyGate(Gate):
    """AHE attribution verdict (the verdict ladder of evolve.py:2283-2303), computed from the
    proposal's predicted_fixes and at_risk fields. HARMFUL edits, and edits with unpredicted
    regressions and no realized fix, are reverted mechanically (AHE leaves rollback to the
    agent; a single lineage needs it here). Everything else is kept, and the verdict appears
    in the next round's history so the optimizer sees its own attribution results."""

    def judge(self, ctx, proposal, before, after) -> GateVerdict:
        fixed, regressed = _flips(before, after)
        predicted = list(proposal.predicted_fixes or [])
        risks = list(proposal.at_risk or [])
        actually_fixed = [t for t in predicted if t in fixed]
        risk_realized = [t for t in risks if t in regressed]
        unattributed = [t for t in regressed if t not in risks]
        n_fixed, n_pred, n_risk = len(actually_fixed), len(predicted), len(risk_realized)
        if n_risk > 0 and n_fixed == 0:
            verdict = "HARMFUL"
        elif n_risk > 0:
            verdict = "MIXED"
        elif n_fixed == n_pred and n_pred > 0:
            verdict = "EFFECTIVE"
        elif n_fixed > 0:
            verdict = "PARTIALLY_EFFECTIVE"
        else:
            verdict = "INEFFECTIVE"
        # unpredicted regressions are harmful even if no declared risk realized
        keep = verdict != "HARMFUL" and not (unattributed and n_fixed == 0)
        return GateVerdict(keep=keep, label=verdict,
                           reason=f"hit {n_fixed}/{n_pred} predicted; {n_risk} risks realized; "
                                  f"{len(unattributed)} unattributed regressions",
                           detail={"actually_fixed": actually_fixed,
                                   "risk_realized": risk_realized,
                                   "unattributed_regressions": unattributed,
                                   "fixed": fixed, "regressed": regressed})
