## Binding constraint: WALL-CLOCK, not probe count
The execution pool (fix_probe / run_episode / ablate) shares a round-wide probe window
that expires on the CLOCK (~3600s) while your count budget is still full. Round 1 proof:
every candidate episode ended with "[INCONCLUSIVE] (not run) -> probe wall-clock for this
phase expired (3600s); probes left: 17" - 17 of 24 real-execution slots destroyed while
hours went into offline simulation. Round 2 proof, subtler: every candidate probed early,
but the two that lost then went 50-75 MINUTES between experiments - the gaps were turns
spent drafting 30KB files offline (often zero bash calls), not analysis. Offline work feels
free; it spends the same clock the experiments need.

Consequences, in order:
1. EXPERIMENT FIRST. By your 10th bash call you must have drafted a change-set in /scratch
   and fix_probed it. (Your own hooks.py blocks offline bash past 10 calls with zero
   experiments - don't fight it, probe.)
2. KEEP THE CYCLE. The winner's pattern in round 2: ~10 probes spaced minutes apart, the
   last one immediately before submit. Your hooks now block bash while the pool has been
   idle >=8 turns or >=16 bash calls, and note at 5 turns. Work in cycles: predict -> probe
   -> read flips -> adjust -> predict. A mechanism you have not seen FLIP a task is a
   hypothesis, no matter how clean the offline simulation looks.
3. DRAFT IN FILES, EMIT ONCE. Build the change-set in /scratch with heredocs/sed/patch;
   emit the full file content exactly ONCE, inside submit_proposal. Re-typing a 30KB file
   in analysis turns is the single biggest wall-clock sink measured on your own episodes.
   Ship small: keep head guards byte-identical, change one file, so screening flips are
   attributable to YOUR mechanism.
4. ONE SHOT EVIDENCE. The mounted `triage` tool answers corpus questions in one call
   (summary/fail tables: repeats, read-only streaks incl. passing max, first-write,
   no-patch, stash usage, env errors; grep: command regex; state: one-shot replay of
   rounds.jsonl - verdicts, val deltas, confirmed vs noise flips, screening, hook stats).
   Do not re-derive a trajectory parser in bash; do not hand-read the journal.
5. VERIFY, THEN CLAIM. Historical predicted-fix hit rate without testing is 0-16%, and
   both kept rounds shipped 0 VERIFIED claims. The Attribute layer's minimize/replay
   experiments are mechanical_fallback noise ("no compression" proves determinism, nothing
   else); real VERIFIED status comes only from fix_probe/run_episode with YOUR edits
   flipping predicted tasks. Submit the byte-identical change-set your last successful
   probe tested, or the verification does not carry over. Label each root_causes claim
   VERIFIED or HYPOTHESIS.
6. SIZE THE EDIT TO THE EVIDENCE. 27,453B/5 files, unprobed, moved validation 18->18
   (2 fixed / 2 regressed); a 32KB single-file probed edit moved 18->19. Prefer one
   mechanism with a measured flip. Prefer the code layer (harness_code/hooks.py) for
   anything mechanical - probes return per-hook call counts and exception tracebacks.
7. NOISE FLOOR. 4/8 recorded flips in round 1 and 3/6 in round 2 failed to reproduce.
   Screening runs on ~12 train tasks and its ranking is itself close to noise (1 vs 0 vs 0).
   Do not build a proposal on a single coin-flip task; do not "fix" a single unconfirmed
   regression.

## Reading the evidence correctly
- The train trajectories in round_N were produced by the PREVIOUS round's head harness.
  They describe what the current head does, not what your draft will do; only probes and
  run_episode with YOUR edits applied measure the draft.
- Attack the largest ALIVE ledger modes; skip EXTINCT ones (the no-write and repeat-loop
  modes collapsed under the current guards - re-measure with triage before re-guarding).
  OMISSION modes need the harness to teach the missing behavior (draft + fix_probe);
  COMMISSION modes need the causal step banned or replaced (validate with substitute).
- Guard thresholds belong ABOVE the passing maximum (triage prints it), never near the
  failing median. ablate PASSING trajectories to find load-bearing winning behavior, then
  distill exactly that; don't spend probes re-confirming near-certain things.

## Non-negotiables
- NEVER submit a hooks.py change without a fix_probe run on it. A 0-flip, 0-exception probe
  means the code ran clean and changed nothing that matters - revise the MECHANISM (layer,
  trigger, threshold), re-probe once, then submit your best change-set regardless.
- Keep edits general: hardwired task ids are rejected by the audit and rot next round.
- Write the FULL file content into every edits[] entry of submit_proposal (a path with no
  content ships an empty file). Drafts left in /scratch are not submitted.
- Call submit_proposal before the budget ends, even at 60% confidence - a submitted
  hypothesis beats an unsubmitted verification. An episode that never submits scores zero.
