# Cross-round memory (persisted via equipment submissions)

## Standing rule: rebuild this file from the journal, not from memory
Every meta-episode, re-read `rounds.jsonl` + `report_cards/round_NN.md` FIRST and overwrite
the "scoreboard" section below. A confidently wrong memory compounds: in R3 my notes
asserted "R2 was HARMFUL, 19->15, reverted" when the journal said EFFECTIVE 19->20 kept,
which would have made me distrust a working mechanism and pick a bad `base_round`.

## Ground truth about the scoreboard (as of the R3 meta-episode)
* R1: val 17->19, MIXED, kept (+4/-2). R2: 19->20, EFFECTIVE, kept (+1/-0, base_round 1).
  R3: **20->19, MIXED, kept (+1/-2)**, reason `keep_all: net 1-2 recorded, not gated`.
  Fixed 1 (`Ljzd-PRO__KToolBox-302`), regressed 2 (`iluvcapra__pycmx-12`,
  `marcosschroh__dataclasses-avroschema-870`), flip_confirm 3/4 reproduced -> the two
  regressions were REAL, not noise. Train sweep 31/110 (R2 sweep was 36/110).
* So: `kept` = not gated. The head after a net-negative round is worse than its parent, and
  the round's `ws_head` is what the next round inherits. Rounds 1-2 = 11,757 then 15,378
  then 18,605 bytes of one hooks.py: layering is monotonically growing and its marginal
  round was negative.
* Prediction channel is worthless so far: R2 {still_F: 2}, R3 predicted_fixes_realized [].
  Do not let a candidate spend turns on predictions.
* Screening: 3 arms on 12 hard failures, scores 0/1/0 in R3 (all 0 in R1, R2), winner
  `self_b`; the winner then netted -1 on val. Tie-breaks are arbitrary -> every arm must
  stand on its own merits, and the screen cannot penalise a regression-prone design.

## The R3 lesson: annotate-only is a regression channel
R3 shipped two annotate-only, tagged mechanisms (`[harness-digest]`: prepend a <=900-char
digest to any long failing test observation; `[harness-import]`: module hint on ANY
observation containing ModuleNotFoundError). Both fired in episodes that were still
progressing. Net: +1 fixed, -2 regressed, both regressions reproduced on rerun.
**Law: fire only at points of no return** (no-tool_use reply = episode ends and grades the
tree; empty diff at exit; byte-identical repeat; empty observation). Those only occur in
episodes already heading to zero -- R1's +4 and R2's +1 lived there. Anything that changes
what a progressing model reads needs a hard per-episode cap, must add rather than reorder
or truncate, and must be probed on 2 currently-PASSING tasks before shipping.

## Measurement facts (hard-won, do not re-derive)
* `hooks_stats['changed']` counts ONLY non-identity returns -> an in-place `after_llm` and
  a hook that RAISED both look absent. Never call a hook 'dead' from that key.
* Ground truth = grep the sweep trajectories for each mechanism's own marker. R2 sweep:
  `[harness]` in 51/110. R3 sweep: 65/110, `harness-empty` in 20/110 -- but the accepted
  R3 head has no such tag: **a rewrite silently deleted a live mechanism** (~18% of
  episodes). Hence the v4 audit that flags any draft dropping a tag the head had.
* `round_NN/trajectories` is the sweep that graded the head you INHERITED, so a tag added
  in round N cannot appear in round N's trajectories. Prove new firing with a probe.
* R3 `hooks_stats.top_errors`: `after_tool: TypeError("expected string or bytes-like
  object, got 'NoneType'")` -- observations are not always strings.
* Free dry-run contract facts: a rebuilt `after_llm` list is re-validated and rejects
  `thinking` blocks (mutate in place, return the same object); unknown `loop` knobs are
  dropped; source cap ~30 KB; banned constructs `open/eval/exec/compile/getattr/setattr/
  globals/locals(`; imports limited to pure-computation modules; only `__init__` dunder.

## Candidate (episode-level) behaviour to design against
* R3 per-arm: self_a 54 bash + 0 probes, submitted turn 21; self_b 64 bash + 2 fix_probe,
  submitted turn 25; self_c ~58 calls, 0 probes, AND burned its first submit at turn 19 by
  passing `edits` as one JSON string. All submitted at turns 19-25.
* Consequence I owned: my v3 verify-gate exempted `turn >= 18`, so it never fired for any
  of the three. v4 moves the window to turn<=22 and accepts verify/bash-hooks_runtime
  evidence, and ships a `verify` tool so verification costs ONE call.
* Pool under-use is chronic: R1 14 used, R2 1/23, R3 2/22. Two of the three arms per round
  spend zero. The fix is cheap tooling + gates in the turn window they actually live in.

## Open questions for the next meta-episode (answer these from the audit files)
1. Did the `verify` / `firecount` tools get called? (report card line "self-written tools
   ... calls this round"; if 0 again, STOP declaring tools and instead run the dry-run
   automatically from before_tool at submit time, or inject the firing table into
   system_prompt -- a tool nobody calls is dead weight.)
2. Did the verify-gate fire, and did any episode end without a submit because of it?
   A gate that costs a submission is worse than no gate -- lower the cap if so.
3. Did any round propose a SUBTRACTIVE change (narrow/delete)? Compare its net vs additive
   rounds. If nobody proposes it even with lens B telling them to, the guidance is not
   concrete enough: name the mechanism to narrow in the next round's search.md.
4. Is the head still net-negative after this round? If two consecutive rounds lose net,
   recommend branching to the last net-positive round explicitly in prompt.md.
5. Watch for the opposite failure: a round so conservative it changes nothing. Net 0/0 is
   not a win; it just spends the round.
