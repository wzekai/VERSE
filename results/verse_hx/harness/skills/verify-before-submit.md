# Before you call submit

- `git --no-pager diff --stat` shows at least one SOURCE file. An empty diff scores zero no matter
  how good your analysis was — and calling `submit` with an empty diff is a guaranteed zero.
- The last thing you ran was a test (or a /tmp reproduction of the issue's example) AFTER your last
  edit, executed with `/opt/conda/envs/testbed/bin/python -m pytest -x -q <file>` — not the default
  `python`, and not a test file you modified.
- The diff matches the ISSUE word for word where it matters: new keyword names, error messages,
  return values, defaults, edge cases named in the issue text. Re-read the issue once and tick each
  sentence off against your diff.
- No `print()`/debug leftovers, no changes under `tests/`, no changes to files unrelated to the issue.
- If the covering test cannot run because the environment is broken (a third-party module really is
  missing), say so in a comment-free minimal patch anyway: a plausible source fix can still score,
  an empty diff never does.
