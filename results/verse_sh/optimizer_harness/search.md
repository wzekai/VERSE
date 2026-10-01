## Your lens this round
Attack OMISSION failures: cases where the executor never produced the needed behavior (missing exploration, missing verification, missing fix). Teach the missing behavior at the right layer.
---
## Your lens this round
Attack COMMISSION failures: cases where a specific wrong action sank the episode (bad edits, destructive commands, wrong submit timing). Ban or replace the causal action at the right layer.
---
## Your lens this round
Attack MECHANICAL failure modes: tool-call formatting, observation noise/truncation, context bloat, error messages the model misreads. Prefer the executable code layer (harness_code/hooks.py) when available — these modes are exactly what instructions cannot reliably fix.
