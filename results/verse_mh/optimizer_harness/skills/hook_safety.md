# Skill: writing executor hooks that don't blow up the arm

The executor's harness_code/hooks.py runs on EVERY episode of EVERY task. One bad
transform is an arm-wide regression; one raise per call is reported in your audit.

## Contract facts (verify in round_NN/executor_source/{executor.py,hooks_runtime.py})
- Episode ends when a reply has no tool_use block and stop_reason=end_turn; the score
  is the accumulated `git diff`. after_llm can append a tool_use block to keep the
  episode alive — append ONLY `text`/`tool_use` typed blocks (a `thinking` block in
  the returned list made the API validator reject the reply in a real incident).
- before_tool returning a (reason, synthetic_result) tuple blocks the command and
  injects the synthetic observation: zero container cost, zero wasted turn.
- after_tool sees the raw observation before the model; the model's own commands are
  visible in before_tool args. State dict persists per episode.

## Safety rules
1. Fail open: wrap every method body in try/except returning the input unchanged;
   a hook that raises degrades to identity AND prints an error in your report card.
2. Act on tight, anchored patterns, never broad keywords; bail out (return input)
   the moment anything unexpected appears (quoted heredocs, pipes, env vars,
   explicit alternative interpreters like `uv`/`poetry`/venv paths — an
   interpreter-rewrite that ignores these sank previously-passing tasks once).
3. Prefer annotating an observation to rewriting a command; prefer blocking a
   never-can-work action (offline fetch) to transforming a maybe-valid one.
4. One-shot messages: gate the gate. A submit-gate must fire once, not loop the
   episode; a keep-alive must count its rescues (cap 2-3).
5. Prove it with fix_probe on 2 failing targets + 1 passing canary. Read the probe's
   per-hook call/changed counts and tracebacks — called 112x/changed 0x means your
   condition never matched (dead code), not success.
6. Ablate before you codify: only distill a behavior into a hook/skill when ablate
   shows it is load-bearing in passing trajectories.
