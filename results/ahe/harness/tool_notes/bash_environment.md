# The task container — facts you can rely on

* **OFFLINE.** No DNS, no PyPI, no HTTP. 67 `pip install`/`pip download` attempts in the
  last sweep, 0 succeeded. If an import fails, switch interpreter (below) or work around
  it — installing is never the answer.
* **Two interpreters.** `which python` is often a conda *base* env that has no pytest and
  none of the project's dependencies (`No module named pytest`). The repository lives in
  its own env, usually `/opt/conda/envs/testbed/bin`. The harness prepends the repo env to
  `PATH` for every command, so `cd /testbed && python -m pytest ...` is correct; if you
  still hit a missing module, use the absolute path
  `/opt/conda/envs/testbed/bin/python -m pytest ...`.
* **Only the repo is graded.** Edits to `/tmp` files, scratch scripts or test files earn
  nothing. `/testbed` source edits are the entire deliverable — check with
  `cd /testbed && git diff --stat`.
* **Hidden tests.** `ERROR: not found: /testbed/...::test_something` = that test does not
  exist in your checkout; the grader supplies it. Stop searching for it and implement the
  contract from the issue text instead.
* **Empty output means "not there".** It is information, not a glitch. Repeating a
  command that printed nothing is the single most common way these episodes die.
