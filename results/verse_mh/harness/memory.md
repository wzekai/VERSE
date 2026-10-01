# Recall card (this container)

- Tests: `cd /testbed && /opt/conda/envs/testbed/bin/python -m pytest <path> -q 2>&1 | tail -40`
  (plain `python`/`pytest` = wrong interpreter, no pytest installed)
- Node ids: quote them, or use `-k`. No network, ever: no pip/conda/curl/wget/clone.
- Score = `git diff` at the end. Draft patch by turn ~10; empty diff = 0.
- No `git stash` / `git checkout -- file` (destroys your patch); no edits to test files
  (the grader replaces them); experiments in /tmp.
- Hidden tests grade the ISSUE, so re-read the issue's sentences against your diff.
