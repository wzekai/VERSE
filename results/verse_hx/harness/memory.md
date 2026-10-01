# Standing facts for this container (verified across tasks)

- Grading = `git diff` at episode end. Empty diff = 0. Reply without a tool call = episode ends now.
- No network: `pip install`, `conda install`, `uv pip install`, `apt-get` always fail. Never try them.
- Repo interpreter: `/opt/conda/envs/testbed/bin/python` (also `/opt/conda/envs/testbed/bin/pytest`).
  The `python`/`pytest` on PATH is a different env without the project's dependencies.
- The grader overwrites `tests/` with its own test files: editing tests earns nothing. Read them for
  the contract; prove behaviour with `/tmp/repro.py`.
- Prefer `grep -rn` / `sed -n 'A,Bp'` over `cat` of whole files: big outputs get truncated anyway.
- Do not re-run an identical command: the harness blocks it. Do not explore past ~20 commands without
  having written a source edit.
