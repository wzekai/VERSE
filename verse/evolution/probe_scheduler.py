"""Fixed tool-selection policies for the ablation on who chooses the experiments.

In VERSE the optimizer decides which tool to run, on which task, and when. These schedulers
spend the same budget through the same ProbeKit but take that choice away: the optimizer never
sees the tools; a policy picks the experiments and their results are given to the optimizer as
read-only evidence.

Two policies:

    random   uniform draws over {replay, ablate} x failed tasks x steps before the episode;
             the submitted draft is checked with fix_probe on randomly chosen failed tasks.
             Tests whether the gain comes from extra execution feedback alone.
    fixed    a hand-written rule, deterministic given the round state: replay the first
             failed tasks (reproducibility), ablate the final action of each further failed
             task (last-step culpability), then run fix_probe on the submitted draft's own
             predicted_fixes. Tests whether a sensible static policy matches the optimizer's
             choices.

Both run in two phases of the optimizer episode:
    pre_episode(ctx)              spends budget before the first turn; the records are
                                  added to the episode prompt
    on_submit(ctx, edits, preds)  runs fix_probe on the draft at submission time (fix_probe
                                  needs a draft, which only the episode produces)

The schedulers call the ProbeKit's own methods, so its budget accounting applies and a
scheduler configuration cannot spend more than the tool-using configuration it is compared
with. pre_episode leaves ON_SUBMIT_RESERVE units unspent for the submission-time fix_probe.
"""
from __future__ import annotations

import random

from .registry import register_component

# fix_probe takes at most 3 target tasks per call (one unit each), so 3 units cover one
# full on-submit call
ON_SUBMIT_RESERVE = 3


def _failed_tids(ctx) -> list:
    """Failed train tasks the tools may run on, in sorted order."""
    allowed = ctx.train_tids or set(ctx.train_outcomes)
    return sorted(t for t, phi in (ctx.train_outcomes or {}).items()
                  if phi is not None and phi == phi and phi < 0.5 and t in allowed)


def _n_steps(ctx, tid: str) -> int:
    tr = (ctx.fail_transcripts or {}).get(tid) or []
    return len(tr)


def _spendable(ctx) -> int:
    return max(0, ctx.probes.remaining() - ON_SUBMIT_RESERVE)


class _SchedulerBase:
    def pre_episode(self, ctx) -> list:
        raise NotImplementedError

    def on_submit(self, ctx, edits: list, predicted_fixes: list) -> list:
        raise NotImplementedError


@register_component("probe_scheduler", "random")
class RandomProbeScheduler(_SchedulerBase):
    """Uniform random experiment choice at the same budget.

    Seeded per round, so a resumed run makes the same draws.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def pre_episode(self, ctx) -> list:
        rng = random.Random(self.seed + int(ctx.round_idx))
        failed = _failed_tids(ctx)
        records = []
        # Bounded by the initial allowance: a failed call can return a record without
        # spending budget, so a loop on the remaining budget alone could spin forever.
        for _ in range(_spendable(ctx)):
            if not failed or _spendable(ctx) <= 0:
                break
            tid = rng.choice(failed)
            steps = _n_steps(ctx, tid)
            if rng.random() < 0.5 or steps == 0:
                records.append(ctx.probes.replay(ctx, tid))
            else:
                records.append(ctx.probes.ablate(ctx, tid, rng.randrange(steps)))
        return records

    def on_submit(self, ctx, edits: list, predicted_fixes: list) -> list:
        if not edits:
            return []
        rng = random.Random(self.seed + 7919 + int(ctx.round_idx))
        failed = _failed_tids(ctx)
        if not failed:
            return []
        targets = rng.sample(failed, k=min(3, len(failed)))
        return [ctx.probes.fix_probe(ctx, edits, targets)]


@register_component("probe_scheduler", "fixed")
class FixedProbeScheduler(_SchedulerBase):
    """A hand-written static policy: replay first, then last-step ablation, then fix_probe.

    The first replay_n failed tasks are replayed (reproducibility), then the remaining failures
    are ablated at their final action until the pre-episode allowance is spent. Tasks are taken
    in sorted order, so the policy is deterministic.
    """

    def __init__(self, replay_n: int = 2):
        self.replay_n = int(replay_n)

    def pre_episode(self, ctx) -> list:
        failed = _failed_tids(ctx)
        records = []
        for tid in failed[:self.replay_n]:
            if _spendable(ctx) <= 0:
                break
            records.append(ctx.probes.replay(ctx, tid))
        for tid in failed[self.replay_n:]:
            if _spendable(ctx) <= 0:
                break
            steps = _n_steps(ctx, tid)
            if steps == 0:
                continue
            records.append(ctx.probes.ablate(ctx, tid, steps - 1))
        return records

    def on_submit(self, ctx, edits: list, predicted_fixes: list) -> list:
        if not edits:
            return []
        failed = _failed_tids(ctx)
        targets = [t for t in (predicted_fixes or []) if t in failed][:3]
        if not targets:
            targets = failed[:3]
        if not targets:
            return []
        return [ctx.probes.fix_probe(ctx, edits, targets)]
