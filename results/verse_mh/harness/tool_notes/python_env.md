# Running anything in this container

```
REPO_PY=/opt/conda/envs/testbed/bin/python      # has pytest + all repo dependencies
cd /testbed && $REPO_PY -m pytest tests/test_x.py -q 2>&1 | tail -40
cd /testbed && $REPO_PY -m pytest "tests/test_x.py::test_name[param]" -q
cd /testbed && $REPO_PY -m pytest tests/ -q -k "keyword" --no-header -x 2>&1 | tail -40
cd /testbed && $REPO_PY -c "import mypkg; print(mypkg.__file__)"
cd /testbed && $REPO_PY -m pip list 2>/dev/null | head -40
```

- `/opt/conda/bin/python` (what `python`/`python3` point at) is the BASE env: no pytest,
  no repo deps. It is never what you want; do not debug its `ModuleNotFoundError`.
- Always `2>&1 | tail -40`: observations are truncated, and a 400-line `-v` dump loses the
  `short test summary info` block that tells you what actually failed.
- `-x` to stop at the first failure, `-k <expr>` to select by name instead of node id.
- Node ids with `[` must be quoted. `-k` avoids the problem entirely.
- The package under test is imported from /testbed - run from the repo root, or
  `sys.path.insert(0, '/testbed')` in a /tmp script.
- A test that needs the network / a missing binary / a GPU cannot be made to run.
  Note it, skip it, and verify with a local repro instead.
