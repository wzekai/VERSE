# Cross-round teacher memory

## Head harness (what is already shipped - do not re-propose it)
Prose layer: `system_prompt.md` + `workflow.md` + `memory.md` + `skills/verify-before-submit.md`.
Code layer: `harness_code/hooks.py` v3 (~32KB, round-2 winner self_a, KEPT). Contents: all
round-1 guards kept byte-for-byte (env-probe prepend; pip/conda-install block; first-test-
file-write block; exact-duplicate block from the 4th execution; read-only block past ~48
consecutive reads with empty diff; after_tool hints; tool-less-reply rescue; empty-submit
gating; on_turn_end notes) PLUS three metered-model helpers: G5 `review_patch` (first non-
empty `submit` becomes a diff-vs-issue completeness review, one metered call, falls back to
a diff-derived sibling grep), G6 clean-context root-cause analyst on the 2nd consecutive red
run (never advises touching tests), G7 `draft_patch` (heading-for-empty-diff episodes get
one harness-drafted minimal edit applied, then the agent refines it). The metered helpers
are NEW and only smoke-tested by screening - treat their interference cost as unmeasured
(see r2 regression below). Guard thresholds must stay ABOVE the passing max (re-measure per
round with triage).

## Scores
- Round 0 baseline: train 33/110, val 18/50.
- Round 1 (prose+code guards, kept): val 18/50 -> 18/50 (2 fixed / 2 regressed); 4 of 8
  recorded flips did NOT reproduce. Edit was 27,453B/5 files with 0 VERIFIED claims.
- Round 2 (hooks v3, kept): train 35/109, val 18/50 -> 19/50 (net 2/1). 3/6 flips
  reproduced; fixed: spdx__tools-python-855, sympy__sympy-27809; REGRESSED and CONFIRMED:
  openforcefield__openff-toolkit-2026. Screening: self_a=1, self_b=0, self_c=0 on 12 train
  tasks - itself noisy; differentiate candidates by MECHANISM, not prose volume.
- Cumulative honesty: three kept edits, net +1 val task over the two kept rounds. Small,
  probed steps only; never bet a proposal on a single flip.

## Failure ledger under the r1 head (measured on round-2 trajectories; re-measure with triage)
- no-patch fails 8/74 (was 16/77); ZERO-WRITE fails 2/74 (was 16/77); rep>=4 fails 2/74
  (was 13/77); read-only streaks: pass max 23 (was 39), fail max 52 -> the 48 threshold is
  now 1.9x the passing max but only 26 below the failing max; tightening toward ~30 is a
  cheap experiment IF zero passing-trace hits.
- 72 of 74 failures WROTE A PATCH and still failed. The ledger's center of mass moved from
  "never produced a patch" to "produced a wrong/incomplete patch". That is why v3 shipped
  the quality helpers - the next job is verifying they FIRE and HELP.
- `git stash` used in 36/109 episodes (15 pass / 21 fail); at least one episode (fsspec__
  universal_pathlib-495, 81 commands, ro-streak 52) ended with the fix stashed and scored
  "no patch submitted (empty git diff)". A stash-hygiene guard is cheap and general.
- env-error episodes ~19/74 still hit `No module named` (PATH python != repo env, no DNS).

## Unshipped mechanisms that LOST screening but were never refuted (merge candidates)
1. Stash-hygiene guard (self_b, r2): track `git stash`; rewrite an episode-ending or
   `submit` turn taken while the stash is applied into `git stash pop` + diff echo; NEVER
   pop mid-episode (would corrupt with/without-fix comparisons). Targets the 8 no-patch
   fails plus stash episodes. Probe: fix_probe on tasks that died with stash applied.
2. Imitation-loop fix (self_c, r2): the tool-less-reply rescue injects a `git diff --stat`
   echo; stuck models imitate the last command, and failing traces end repeating
   harness-injected no-op echoes. The rescue should hand a PATCH, not a status echo.
   Verify from the round's trajectories whether the head still does this, then fix + probe.

## Process facts about MY teaching episodes (round 2, audited)
All three candidates probed inside ~6 min and submitted exactly once (v1 contract worked).
Winner self_a = probe CYCLE (10 probes minutes apart, last one right before submit). Losers
self_b/self_c stalled 50-75 MINUTES between experiments - the gaps were generation turns
drafting 27-36KB files offline (often ZERO bash calls), not analysis scans. So pacing is
now metered in TURNS since last experiment. v2 teacher hooks: bash blocked while the pool
has been idle >=8 turns or >=16 bash calls (after the first experiment), note at 5 turns,
post-submit bash cap 12; `triage` gained mode=state (one-shot journal replay) and a
JSON-transcript fallback. Residual risk: drafts belong in /scratch via heredoc/sed, full
file content emitted ONCE inside submit_proposal.

## Process failure to never repeat (round 1, fixed but watch)
Round 1: 200+ bash calls/episode, 0 experiments, 17 of 24 execution slots destroyed unused
when the wall-clock died. Round-2 lesson in the other direction: the CLAIMS pipeline still
returned 0 VERIFIED / 3 HYPOTHESIS / 2 REFUTED - minimize/attribute experiments are
mechanical_fallback "no compression" noise and prove nothing; real VERIFIED status comes
only from fix_probe/run_episode with YOUR edits flipping predicted tasks.

## Open questions for round 3 (answer with probes, not simulation)
1. Did G5/G6/G7 actually FIRE in the round-3 sweep (hooks_stats / HARNESS_CODE_SUMMARY per
   trajectory), and did any pass episode get slowed or blocked by them? A metered helper
   that never fires is dead weight; one that fires on passing traces is a regression risk.
2. Is the openff-toolkit regression attributable to a metered helper (review/analyst giving
   bad advice)? If yes, make that helper strictly advisory (cannot rewrite the submit).
3. Ship stash-hygiene and/or the patch-handing rescue (above) as SMALL, probed diffs that
   keep the rest of hooks.py byte-identical.
4. Read-only threshold: with passing max now 23, test tightening 48 -> ~30 offline first;
   ship only with zero passing-trace hits, then confirm with a probe.
