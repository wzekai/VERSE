# Verify the CONTRACT, not the old tests

The graded tests are hidden and are written against the behaviour the ISSUE promises. The
tests already in the repo passed before your change, so a green run of them proves
nothing. In the last sweep 21 of 26 recoverable wrong-patch episodes ran the very test
file the grader used, saw it completely GREEN, and still failed: the graded test is an
updated version of that file, so local green is the normal state of a broken patch.

Work loop, every episode:

1. **Read the issue as a contract.** Write down the literal promises: module, function and
   argument names, defaults, exception types, message wording, and the behaviour on the
   edge case the reporter showed. The harness pins these into a `[CONTRACT CARD]` block.
   When the issue lists several concrete failures (a stack trace, N validation errors, a
   table of cases), that list IS the work order: one sub-fix per item.
2. **Locate, <= 12 commands.** `grep -rn "<name from the contract>" /testbed --include=*.py`,
   then `sed -n 'A,Bp' file` on the candidates. A command that returned nothing is never
   re-run. If the issue names a function, patch THAT function, not a neighbour.
3. **Edit the source** (small python heredoc replace, or `sed -i`), then
   `cd /testbed && git diff --stat` to prove it landed. Keep it small: passing patches
   median 1 file and 4 added lines.
4. **Prove the new behaviour.** Write `/tmp/contract_test.py`: copy the example out of the
   issue and assert the promised output exactly -- value, ordering, whitespace, rounding,
   exception type. Run it with the repo interpreter. Never assert what the current code
   does; assert what the issue says it should do. Keep it in /tmp.
5. **Call `verify` once.** It is deterministic -- nothing in it was written by a model --
   and it reports: py_compile of your files, the covering tests, your repro, the issue
   names still absent from the repo, and for every red test an A/B re-run WITH YOUR PATCH
   REVERTED. Read the A/B block as the decisive line:
   * `BROKEN BY YOUR PATCH ... green at HEAD` -> your own regression, the grader counts
     it as a failure. Narrow the edit or restore the behaviour; never submit with this.
   * `already red at HEAD` -> not yours; do not spend turns on it.
   * everything green -> your patch is unproven, not correct: re-read the issue for the
     edge/invalid case your code still misses.
6. `submit`.

Hidden-test node ids (`ERROR: not found: tests/test_x.py::test_y`) are not in your repo:
never grep for them, never conclude the issue is "about" a test you cannot find. The issue
text is the specification.
