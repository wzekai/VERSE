# Teacher memory (cross-round) — condensed, keep updated every meta-round

## Scoreboard
- r0 head: val 16/50, train 31/110.
- r1 (kept, MIXED): env-normalising + patch-protecting hooks.py + prompt set -> val 17/50, train 31/110. Net +3 fixed / -2 regressed; only 5/9 flips reproduced on rerun (confirmed fixed: 2 of the claimed; the litellm-style REGRESSION confirmed real). LESSON: single-run flips ~coin flips; transforming hooks carry real regression risk.
- r1 screening mechanics: 3 candidate episodes with the 3 search.md lenses; each ~55-60 turns; winner gets a second (final) episode; execution pool (fix_probe+run_episode) shared round-wide (~24 slots; 11 used). Winner r1 = lens 1 (OMISSION).

## Current head harness (r1 state — do NOT re-read from scratch)
harness_code/hooks.py (~26KB, executor-side), error_rate 0.0 over 50 eps:
- system_prompt tail: decisive facts (repo env interpreter /opt/conda/envs/testbed/bin/python; NO network; score=git diff; quote pytest node ids; hidden tests grade the ISSUE).
- before_tool: refuses remote fetches w/ synthetic "no network" obs; blocks git stash/restore/clean/reset once a diff exists; duplicate read-commands served from cache; submit gate (demands a whole-test-file run before submit).
- TRANSFORMS (regression suspects): rewrites bare `python`/`pytest` invocations to repo env; quotes parametrized node ids. These fixed classes but plausibly sank the 2 confirmed regressions (repos with legit other envs; mangled rewrites). Any edit here: probe currently-passing canaries.
- after_tool: env-error annotations (no-pytest / offline / ERROR-not-found); read-cache.
- after_llm: revives episode-ending replies that are markup-only or actionless (appends a tool_use block; ONLY text/tool_use block types — thinking blocks broke the validator in r1, fixed).
- on_turn_end: empty-diff checkpoint notes (198 changes/50 eps — pacing OK). loop: keep_full_turns 16, nudge_no_tool text.
- Markdown: system_prompt.md, workflow.md, memory.md (recall card), tool_notes/python_env.md, skills/verify_before_submit.md.

## Corpus facts (r1 triage, 110 tasks, use corpus_triage tool for fresh numbers)
- 31 pass / 79 fail. Flags: base_python_no_pytest ~41-50, pip_try_offline 28, net_attempt 26, green_but_graded_fail 18, never_ran_tests 17, mangled_nodeids 16, no_edit_seen 14.
- Classes: FAILED-with-a-patch ~40 (CORRECTNESS gap — biggest untouched, next frontier), NOT_RUN ~22 (several UNWINNABLE: the verifier's own pytest dies — offline uv, git-apply conflict, parametrized id truncated in eval script; don't chase), NO_PATCH 13, REGRESSION 3.
- 16 failures had fully green last pytest run -> graded by hidden tests matching issue wording (names/messages/edges). "Existing tests pass" is not the bar.
- Passing eps (mean ~29 turns): grep-locate -> sed/python edit -> repo-env pytest on touched file -> whole dir -> git diff -> submit. 0/31 skipped tests; 17/79 fails never ran tests.

## Executor loop mechanics
- Episode ENDS on a reply with no tool_use + stop_reason=end_turn -> graded on accumulated git diff. after_llm can append tool_use to keep alive.
- before_tool tuple return = block + synthetic observation: no container command, no turn wasted.
- self.llm costs a turn. obs are truncated: teach `2>&1 | tail -40`.
- Duplicate round dirs: trajectories at round_NN/trajectories (front-matter eval_passed, n_steps, last_error; steps as ```bash blocks + observations). Executor source at round_NN/executor_source/{executor.py,hooks_runtime.py}. Current head also mirrored at workspace/.

## Method rules (hard-won — obey)
1. fix_probe the DRAFT BYTES before submit: 2 predicted-fix targets + >=1 REGRESSION CANARY (task the current head passes). Submit byte-identical to the probed set. r1's biggest miss: never probed a canary -> -2 regressions.
2. A refuted probe downgrades specific task claims, not the mechanism (empty-diff -> real-patch = mechanism verified even at 0 flips).
3. Max 2 probes per hypothesis, then move on; never burn the shared pool (>4/episode starves the final episode).
4. Never end without submit_proposal. An unsubmitted episode = 0.
5. ~50% of single-run flips are noise; the report card reruns flips — build claims that survive twice.
6. claims.json wants experiments; minimize/ablate on FAILING traces rarely flips — ablate PASSING traces to find load-bearing behaviors and distill them into skills.
7. Don't raise obs_cap/max_turns for free — check context pressure first (r1 left loop mostly default).

## Open leads for next round
- CORRECTNESS (40 FAILED-with-patch): help the executor hold the diff against the issue's assertions — e.g. an extra tool that diffs requirements vs patch (metered self-review via env.llm), or a run_tests tool that also prints the touched module's full test results compactly. Verify via fix_probe (correctness flips are noisy — probe twice if pool allows).
- Regression-proofing the head: guard transforms (skip rewrite when the command mentions an explicit different interpreter/venv/uv/poetry; skip gate if a prior submit was blocked already).
- NO_PATCH survivors: keep-alive exists; check whether the rescue commands actually run (after_llm changed 0/112 calls in probes — rescue FIRES but the appended command may not be executed? investigate before adding more).
- Verifier-NOT_RUN tasks are dead budget: identify from verifier output and blacklist mentally (they rotate; re-identify each round from fresh output, not from stale ids).
