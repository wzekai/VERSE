# Turn budget discipline

The episode is short and wall-clock capped; failing episodes wander, passing ones lap
the loop early and then iterate on evidence.

| turns | what must have happened |
|-------|--------------------------|
| 1-8   | located the responsible function + read the tests that cover it |
| ~10   | a first draft patch is ON DISK (`git diff --stat` non-empty) |
| 10-25 | run `/opt/conda/envs/testbed/bin/python -m pytest <test file> -q`, read the first failure, fix, re-run |
| last  | `git diff` reviewed against the issue's requests -> `submit` |

Rules of thumb:
- One command per turn, the shortest one that produces new information.
- Never run the same read-only command twice; the output has not changed.
- Never `cat` a whole large file: `sed -n 'A,Bp'` the region you need.
- Never read installed copies under site-packages - edit the checkout in /testbed.
- If two consecutive commands teach you nothing, stop analysing and write your best
  hypothesis into the source: a wrong patch can be corrected by a test failure, an
  empty diff cannot be corrected by anything.
