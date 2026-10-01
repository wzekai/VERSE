# Skill: working in a Python task repo

**Find the right interpreter once, then reuse it.** The system `python` usually has none of the
project's dependencies; the repo lives in a dedicated env:

```bash
ls /opt/conda/envs 2>/dev/null; ls -d /testbed/.venv 2>/dev/null
/opt/conda/envs/testbed/bin/python -c "import <package>; print(<package>.__file__)"
```

Use that path verbatim for every later run: `PATH/TO/python -m pytest ...`,
`PATH/TO/python -c "..."`. If an import still fails, the harness prints a `GOODPY <path>` probe
line — switch to it and stay on it. Do not `pip install` anything: the sandbox is offline.

**Run tests cheaply.** `-p no:cacheprovider -q` plus a file or `-k` selector, and pipe through
`| tail -40` so the summary survives. A `pip list`/`conda env list` inventory costs a turn and
tells you almost nothing; a successful import of the package under test tells you everything.

**Read code cheaply.** `grep -n` for the symbol, then `sed -n 'A,Bp'` for the range. If you must
see a whole function, ask for its line range, not the file.

**Reproduce in one command** instead of writing a script file:
`PATH/TO/python -c "...short repro..."` — it shows the pre-fix failure and, after the edit, the fix.

**Regressions count.** After the fix, run the whole test file you touched (not the whole suite);
a previously passing test that now errors is a failed task even if the new behaviour works.
