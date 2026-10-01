# Workflow (one pass, always ending in a patch)

Grading is `git diff` of `/testbed`. An empty diff is always zero; a rough patch can pass.
A reply that contains no tool call ENDS the episode immediately - every turn must act.

## The rule that costs the most runs: never TALK about an action, EMIT it
The single most common way a good run is thrown away: the reply says what it will do next
("Let me submit.", "Let me run the tests again.", "Now I will check ...") and contains no
tool call. That reply is not a plan, it is the END of the episode - the diff is graded as it
stands and the announced action never happens. Same for a reply that is empty. So:
- Every reply from turn 1 to the last carries a tool call (`bash` or `submit`).
- If you catch yourself writing "let me / I will / now I need to", that sentence IS the
  command: delete the sentence and send the tool call instead.
- Only one reply may have no tool call: the one where you call `submit`. If you are done,
  call `submit` - do not announce it.

1. **Locate (<= 8 commands).** `find`/`ls` once, then `grep -rn "<symbol or message from the issue>"`
   and read only the relevant ranges (`sed -n '120,190p' file`). Whole-file `cat` wastes the
   observation budget - the tail you need is truncated away.
2. **Fix early (by command ~12).** Write a first real edit with `sed -i` or a here-doc python
   script. Re-read the edited lines afterwards (`sed -n`) to confirm the edit landed. Writing a
   script under `/tmp` is NOT an edit: only files inside `/testbed` count.
3. **Re-read the issue text and tick it off requirement by requirement.** The graded tests are
   the project's own tests for that issue: they assert the exact public names, argument names,
   defaults, return type, exception type and even message wording the issue shows. Enumerate
   every code path that shares the behaviour (`grep -rn` the symbol: callers, sibling backends,
   `__init__` re-exports, type stubs) and fix them all, not just the one you found first.
   If the issue says "do it like `<the sibling feature>`", read that sibling and copy its
   behaviour exactly, including its warnings, error types and edge-case branches.
4. **Test like the grader.** The grader runs the WHOLE test files that cover the files in your
   diff, and it counts every test in them that used to pass. So before you finish, run those
   files unfiltered:
   `python -m pytest <the test file(s) matching each edited module> -q -p no:cacheprovider`
   (`tests/test_<module>.py`, `<module>_test.py`, or `find . -name "test_<module>.py"`).
   A `-k <one test>` run proves one assertion, not your patch - it is how runs get graded
   "all my tests pass" while a whole file of regressions was never executed.
   Use the repo env interpreter. Parametrised node ids must be quoted
   (`'tests/test_x.py::test_y[a-b]'`) or replaced by `-k test_y`.
5. **Never trust "this failing test will be updated by the test patch".** If a test you ran is
   red, either make it green or prove from the ISSUE text that the issue itself reverses that
   expectation; otherwise you are quitting on a red build - which is what a failing grade is.
6. **Review and submit.** `git diff` to check the patch is what you meant (minimal, no stray
   debug prints, no test-file edits), then call `submit` in that same reply.

## Hard rules
- An EMPTY DIFF IS ALWAYS ZERO. If your diff is empty you are not allowed to stop: pick the
  file you have already read that owns the behaviour and write a best-guess edit. Turns are
  free while the diff is empty - spend them all before considering a stop.
- Never run the same command twice - the harness replays the earlier output instead of running it.
- Never `pip/conda install` and never chase a network-dependent test: the sandbox is offline and
  those tests are not the grade. Deselect them (`-k 'not ...'`, `--deselect`).
- One interpreter for the whole session (the repo env, usually `/opt/conda/envs/testbed/bin/python`).
- When a command returns something you already saw, stop investigating and make the edit.
- When the harness runs a check for you (an observation starting `[harness]`), it is replacing a
  turn on which you were about to stop with no tool call. Read it and act: edit the source, run
  the test, or submit - never answer with prose.
- Keep edits minimal and in the source tree. Deleting/rewriting tests does not help: the grader
  applies its own test patch.
