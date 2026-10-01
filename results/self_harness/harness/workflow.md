# Workflow (read before the first command)

You are graded on ONE thing: the git diff you leave in /testbed, scored by the maintainers' own
hidden test file. Nothing else counts.

1. **Turn the issue into a checklist before you edit.** One line each: what must be accepted, what
   must be returned or printed, which exception must be raised, which exact names and defaults the
   issue uses. Those lines are the spec. The hidden tests check all of them; only the repo's old
   tests are visible to you.
2. **Locate, then edit the source.** grep / find / cat to the module, then rewrite the file with a
   python heredoc (`open`, `replace`, `open(path, w)`) or `sed -i`. Prove it with `git diff`. Never
   finish with an empty diff; never edit a test file (the verifier overwrites them).
3. **Mirror the change.** If the touched name is defined in more than one place (another backend, a
   sync/async twin, a base class, a subclass), apply the behaviour in every one of them - the hidden
   file may exercise the copy you did not touch.
4. **Run the module that COVERS your change**, not one file you happened to notice:
   `grep -rl --include=*test*.py <module> .` then
   `/opt/conda/envs/testbed/bin/python -m pytest <that module> -q 2>&1 | tail -40`. A green run of
   the OLD tests proves only that you broke nothing. Run the whole file, never a parametrised node
   id. Never re-run a command you already ran - the harness blocks it.
5. **Answer the harness review, do not just read it.** On your first submit the harness calls
   `precheck`: it builds an executable check from the ISSUE text (`/tmp/spec_test.py`), runs it, and
   prints it above an independent reviewer's bullets, the test modules importing your changed files,
   and every other definition of the names you touched. Treat a faithful red spec check as a failing
   graded test: fix the source until it is green. Dismiss a check or a bullet only if you can point
   at the issue line that disproves it - "the old tests pass" is not a dismissal.
6. **No network.** Never pip/conda/uv install, git clone or wget. A missing import means you used
   the wrong interpreter, not that a package is missing.
