# Results

The harnesses selected in the paper's main experiments on SWE-rebench, with their per-task test
outcomes. `python results/make_table.py` prints the results table from these files.

| Directory | Setting | Config |
|---|---|---|
| `blank/` | initial harness h0, no evolution | -- |
| `meta_harness/` | Meta-Harness | `baselines/meta_harness` |
| `verse_mh/` | Meta-Harness + VERSE | `self_teacher/verse_mh` |
| `ahe/` | AHE | `baselines/ahe` |
| `verse_ahe/` | AHE + VERSE | `self_teacher/verse_ahe` |
| `self_harness/` | Self-Harness | `baselines/self_harness` |
| `verse_sh/` | Self-Harness + VERSE | `self_teacher/verse_sh` |
| `harnessx/` | HarnessX | `baselines/harnessx` |
| `verse_hx/` | HarnessX + VERSE | `self_teacher/verse_hx` |

## Files of one setting

```
harness/                    executor harness of the round selected on validation
  system_prompt.md, workflow.md, memory.md, skills/, tool_notes/   prompt files
  harness_code/hooks.py     code hooks on the executor loop (if any)
optimizer_harness/          (VERSE settings only) the optimizer's own harness in that round
  prompt.md, search.md, notes.md, skills/                          prompt files
  teacher_code/hooks.py     tools and hooks on the optimizer loop (if any)
run.json                    config, models, selected round, validation result, checksums
test_in_distribution.json   test outcomes, in-distribution test set (108 tasks)
test_ood.json               test outcomes, out-of-distribution test set (107 tasks)
```

The harness files are kept exactly as they were evaluated, including text written by the
optimizer. Some of them use the code's names: *teacher* = optimizer, *student* = executor,
`fix_probe` = the verification tool, `ablate`/`substitute` = the perturbation tool.

In a test file, `outcomes` maps each task to `[r1, r2, r3]`: 1 if the task was solved in that
of the three test evaluations. `accuracy_mean` and `accuracy_sem` are the mean and standard
error over the three evaluations. `excluded_tasks` are not scored (their reference solution
fails in the grading environment).
