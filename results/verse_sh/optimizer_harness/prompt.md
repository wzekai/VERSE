## Experiment strategy (spend your budget where the answer is uncertain)
1. fix_probe is your highest-value tool: BEFORE submitting, draft your edits and fix_probe them on 2-3 tasks you predict they fix. Historical predicted-fix hit rates without testing are 0-16% — a verified draft is the single biggest upgrade you can make to a proposal. Iterate: 0 flips -> revise -> probe again, but AT MOST twice; then submit your best change-set regardless. A refuted probe downgrades specific predicted_fixes, it does not veto the proposal — probes test 1-3 noisy tasks, and probe-refuted proposals have measured best-of-arm on the full validation sweep. NEVER end an episode without submitting: an unsubmitted episode scores zero and wastes the whole round. Evidence attaches to the EXACT change-set probed: submit the same edits your last successful probe tested (byte-identical), or the verification does not carry over to your submission.
2. The report's Attribute layer (when present) already carries EXECUTED minimal cores with an omission/commission classification and a failure-mode ledger — route on it: OMISSION modes need the harness to teach the missing behavior (draft the edit, validate with fix_probe); COMMISSION modes need the causal step banned or replaced (validate the replacement with substitute). Attack the largest ALIVE/GROWING ledger modes first; do not re-attack EXTINCT ones.
3. ablate PASSING trajectories to prove which winning behaviors are load-bearing, then distill exactly those into skills (successes live in trajectories/ too).
4. Do NOT spend budget re-confirming what is already near-certain (replaying a deterministic failure, ablating single steps of failed traces: these almost never flip). Verification tells you WHERE to be confident, not to shrink your ambition: propose the full set of edits the evidence supports.

## Code-edit discipline (you have fix_probe — use it)
Code edits are high-variance: one wrong after_tool can truncate every observation; one
right sanitize mechanically kills a whole failure class. NEVER submit a hooks.py change
without a fix_probe run on it: the probe mounts your draft code on real episodes and
returns per-hook call counts and exception tracebacks alongside the flip results. A
0-flip, 0-exception probe means your code ran clean but changed nothing that matters —
revise the mechanism, not just the syntax.
