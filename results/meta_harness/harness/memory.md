# Cross-task lessons (general, no task ids)

- The environment is a conda/venv image at `/testbed`, offline, graded by `git diff` + the
  project's own hidden tests. Two facts about the environment matter more than anything else:
  the interpreter that has the project installed, and the test file that covers the code you edit.
- **How runs actually get thrown away.** The episode ends the moment a reply has no tool call,
  and the diff at that instant is the final answer. The most common self-inflicted zeros are:
  (a) a reply that ANNOUNCES the next action ("Let me submit.", "Let me run the tests") instead
  of emitting it - the announced action never runs;
  (b) an empty reply (thinking without an action) which also ends the run;
  (c) no edit at all, after any amount of good analysis;
  (d) repeating reads instead of editing.
  Countermeasure for all four: every reply carries a tool call, and if you cannot name the
  exact fix, write your best-guess edit anyway - a rough patch can pass, an empty one cannot.
- **"All my tests pass" is not a finish line.** The grader re-runs the entire test files that
  cover your diff (plus every test in them that used to pass), and it also runs NEW tests
  written from the issue text. So: run the covering test FILES unfiltered once before you stop,
  and re-read the issue line by line against your `git diff` - every name, default, exception
  type and message the issue shows must appear literally in the patch, and every sibling code
  path (sync/async, other backends, base classes, re-exports) must get the same change.
- A failing test you chose to ignore is a decision, not an accident. Justify it from the issue
  text or fix it.
- Shell traps that look like code bugs: unquoted parametrised node ids come back as
  `ERROR: not found` (quote them or use `-k`), and `ModuleNotFoundError` means the wrong
  interpreter, not a missing dependency (never install; use the repo env python).
- Failing tests that need the network/live credentials are not your problem - deselect them and
  move on. Everything you can control is the source diff.
