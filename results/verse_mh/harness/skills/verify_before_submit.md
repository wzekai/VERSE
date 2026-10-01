# Before you call submit

1. `cd /testbed && git diff --stat` - empty means you are about to score zero. Write the
   fix first, even a partial one that captures the intent.
2. Run the WHOLE test file (or directory) of every module you edited, not just the one
   test you were aiming at: regressions hide next door and cost more than the feature
   point. `/opt/conda/envs/testbed/bin/python -m pytest tests/test_touched.py -q 2>&1 | tail -40`
3. Hold the diff against the ISSUE, sentence by sentence. Each "X should Y", "must not
   Z", "returns/raises W" is a graded assertion. Ask of each: which line of my diff
   implements it? Missing ones are the usual reason a "green" run still fails grading.
4. Check the shape of the change:
   - did you change only source (no test files, no stray files, no `print`)?
   - did you handle the neighbouring cases (other callers of the function, the same
     pattern in a sibling class, the `None`/empty/`-1` edge)?
   - is the public name/signature/message exactly what the issue asked for, or just
     something equivalent? Hidden tests match names and messages.
5. `submit`. Do not polish further: each extra turn risks losing the thread, and the
   graded test file is not one you can read.
