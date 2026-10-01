## Your lens: QUALITY OMISSION (a patch existed and was still wrong)
Last round 72/74 failures wrote a patch yet scored 0: the graded tests are not in the
checkout, so locally-green != issue-solved. The head's NEW metered helpers (G5 review_patch,
G6 red-run analyst, G7 draft_patch) target exactly this but have never been evaluated end
to end. First read the round's hook ledger/hooks_stats: did they fire, on which traces, and
did any PASSING trace get slowed or blocked? Then attack what is still missing: reproducing
the issue text as one executable check, running the WHOLE covering test file, grepping
sibling implementations before naming things. Rules: `triage` once (state + summary), draft
in /scratch and fix_probe on 2-3 previously-failed-WITH-PATCH tasks by your 10th bash call;
if 0 flips change MECHANISM (trigger/threshold/layer), not wording; keep every existing
guard byte-identical; label claims VERIFIED/HYPOTHESIS; submit_proposal before the clock.
---
## Your lens: COMMISSION (one action sank a good fix)
Attack single destructive actions at the mechanical layer. Prime target measured last round:
`git stash` used in 36/109 episodes; an episode that ENDS or submits with the fix still
stashed auto-zeros ("no patch submitted") - an existing fix destroyed at the last second.
Other modes: editing test files (grader discards them), edits that revert themselves, the
48-command read-only spiral. Mechanism: track stash state; rewrite an episode-ending/submit
turn taken while the stash is applied into `git stash pop` + diff echo - never pop
mid-episode (it would corrupt legitimate with/without-fix comparisons). Ban or REPLACE the
causal step via before_tool (block or synthetic_result) and validate with substitute/
fix_probe, not reasoning. Prove from the hook ledger it never fires on a passing trace.
`triage` once; probe by the 10th bash call; keep existing guards byte-identical; label
VERIFIED vs HYPOTHESIS; submit before the clock, always.
---
## Your lens: MECHANICAL (the harness itself breaks episodes)
The observation layer is lossy: long outputs keep head+tail, so an APPENDED hint is elided
(prepend hints); the tool-less-reply rescue injects a `git diff --stat` no-op that stuck
models imitate - failing traces end by repeating harness-injected status echoes, and an
episode ending in a no-op echo scores 0. The rescue should hand the model a PATCH, not a
status. Also: metered helper calls spend episode budget and can mislead (one confirmed val
regression rode with the round-2 helpers). Fix at the observation/submit surface in
harness_code/hooks.py; wrap every hook body in try/except (a raising hook degrades to
identity - a silently dead hook is worse than none). Falsify cheaply: probes return per-hook
call counts and tracebacks. Rules: `triage` once (state mode first); draft in /scratch, fix_
probe by the 10th bash call; a 0-flip 0-exception probe means your code did nothing - revise
the mechanism, then submit your best change-set regardless. Never hardwire task ids.
