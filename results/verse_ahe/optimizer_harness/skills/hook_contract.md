# Skill: the hooks.py contract, and how to verify hook code for FREE

`harness_code/hooks.py` is loaded into the executor loop and every hook call is guarded:
a raising or badly-typed return degrades to IDENTITY for that call and is written to a
per-episode ledger that shows up in the round journal (`hooks_stats.top_errors`) and in
fix_probe output. A hook that is absent from `hooks_stats.changed` never fired once -- the
single most common way a mechanism "does nothing" while the author believes it works.

## Hard contract (read out of the runtime source, not guessed)
* `system_prompt(assembled) -> str`.
* `before_llm(msgs, state) -> list`, `after_llm(content, state) -> list`:
  - **Returning the SAME list object you were given skips re-validation.** Returning a
    NEW list triggers validation that accepts ONLY `type in {text, tool_use}`. This
    executor emits `thinking` blocks, so `return [b for b in content ...] + [note, call]`
    raises `bad return: unknown block type 'thinking'` and the whole hook is discarded.
    Correct shape: `content.append(note); content.append(call); return content`.
  - Only the first 12 blocks of a rebuilt list survive; blocks placed after a `tool_use`
    are auto-reordered before it (Messages-API constraint) and a reorder is ledgered.
  - Do not reorder or delete the model's own tool_use blocks: the loop reads them.
* `before_tool(name, args, state)` -> `dict` (possibly rewritten args) | `str` (block, the
  str is the reason) | `(reason, synthetic_result)` (block and feed `synthetic_result` to
  the model AS the observation). Blocking a command the model needs is the #1 source of
  regressions: prefer annotating in `after_tool` to blocking, and if you block, hand back
  something that keeps the work moving (cached output + one next action).
* `after_tool(name, args, obs, state) -> str`. Append, never truncate blindly: a too-short
  `obs_cap` or an aggressive digest silently removes the evidence the model needs.
* `extra_tools()` / `run_tool(name, args, env, state)`: new tools; `env` only reaches the
  task container / metered `self.llm` (one llm call = one turn of the budget).
* `on_turn_end(turn, state) -> str|None`: injected as a `[harness note]` text block on the
  next user turn. Long notes pushed every turn degrade passing episodes (measured: a
  round whose mechanism injected prose into most observations netted -4 and was reverted).
* `loop = {...}` knobs only: `max_turns` (only shrinks), `obs_cap` 1024-12288,
  `keep_full_turns` 4-32, `spam_streak` 3-10, `nudge_no_tool`, `nudge_bad_markup`
  (<=400 chars). Unknown keys are dropped AND ledgered -- a typo is visible, but useless.
* Static audit rejects: imports outside {re, json, math, string, textwrap, difflib,
  collections, itertools, functools, heapq, bisect, statistics, copy, fnmatch, shlex};
  `open/eval/exec/compile/input/getattr/setattr/type/super/...`; dunder attribute access;
  `__dunder__` method definitions except `__init__`; async; global/nonlocal; size caps
  (executor default 32000 chars). A violation can bounce the whole proposal.

## The free verification loop (no pool, no wall-clock, ~1 second)
Probes are scarce (a shared pool per round, and the phase dies on a 3600s wall-clock).
Contract bugs cost a whole sweep; unit tests cost a shell command. Do this BEFORE any
fix_probe/run_episode:

```bash
R=$(find . -name hooks_runtime.py 2>/dev/null | head -1); D=$(dirname "$R")
cp your_draft.py /scratch/draft.py
python3 - "$D" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
import hooks_runtime as H
src = open('/scratch/draft.py').read()
print('audit:', H.audit_hooks_source(src, max_chars=64000))
rt = H.HookedRuntime(src, max_chars=64000)
# adversarial inputs -- the shapes that actually break hooks in production:
rt.system_prompt('BASE')
rt.before_llm([{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}], 1)
rt.after_llm([{'type': 'thinking', 'thinking': 'x'},
              {'type': 'text', 'text': 'done'}], 2)               # NO tool_use + thinking
rt.after_llm([{'type': 'tool_use', 'id': 'a', 'name': 'bash',
               'input': {'command': 'ls'}}], 3)                    # trailing block order
rt.before_tool('bash', {'command': 'pip install foo'}, 4)
rt.after_tool('bash', {'command': 'ls'}, '', 5)                    # empty observation
rt.after_tool('bash', {'command': 'ls'}, 'E' * 40000, 6)           # huge observation
rt.on_turn_end(7)
print('errors:', rt.errors)
print('changed:', rt.changed)
PY
```

PASS = `audit: []`, `errors: []`, and every hook you expect to fire shows up in `changed`
for the scenario that should trigger it. If your hook never appears in `changed`, the
mechanism cannot fire (wrong tool name, wrong state key, wrong turn window) and no amount
of probing will make it do anything.

## Cost model for the rest of your verification budget
1. Free: the dry-run above. Repeat it after every edit. (Unlimited.)
2. Cheap: ONE `run_episode`/`fix_probe` on a SHORT failing episode (a dozen or two
   commands) purely to prove no crash on a real container. Big repos / long loops blow
   the wall-clock and buy nothing.
3. Cheap and mandatory: ONE probe on a task that CURRENTLY PASSES, to bound the
   regression blast radius. Report it in `at_risk`.
4. Don't: re-probing the same long failure (0-flip base rate), replaying deterministic
   failures, ablating single steps of failed traces.
