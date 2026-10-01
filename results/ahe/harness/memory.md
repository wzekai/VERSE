# Harness memory (episode-independent facts that repeatedly paid off)

**Container (measured over 400+ episodes).**
- Offline. `pip install` / `pip download` never succeed (67 attempts / 0 successes).
- The repo's own interpreter is `/opt/conda/envs/testbed/bin`; the default `python` can be
  a conda base env with no pytest. The harness prologue prepends the repo env for you, so
  `cd /testbed && python -m pytest <file> -q` is right.
- Hidden tests: `ERROR: not found: <file>::<test>` means the graded test is not in your
  repo. Grading also reports `not-run` when the runner's node-id selection mangles ids that
  contain commas -- neither is your bug. Implement the issue text instead.

**What the grader rewards (measured).**
- An empty `git diff` grades 0. A plausible patch always beats a perfect analysis.
- **A green local run is not evidence.** 21 of 26 recoverable wrong-patch failures ran the
  grader's own test file, saw it green, and still failed: the hidden test is an UPDATED
  version of that file and exercises behaviour your patch may not add.
- **Only a comparison against HEAD proves anything.** `verify` re-runs every red test with
  your patch reverted. `BROKEN BY YOUR PATCH (green at HEAD)` is a regression you caused --
  5 graded failures in the last sweep were exactly that, and the grader marks them
  `(regression)`. A test that was already red at HEAD is not yours: ignore it.
- Model-written "independent checks" are noise, not evidence: a FAIL verdict predicted the
  outcome at 34% and a PASS at 36%. The harness no longer runs them. Trust only numbers
  produced by running the code both with and without your patch.
- Implement the promised API **literally**: `verify` lists the issue names that do not
  exist in the repo yet -- those are the names the hidden test will call.
- One-sided patches are a classic miss: sync/async twins, base classes, other backends.
- Focused beats broad: passing patches median 1 file / +4 lines, failing ones +11.5 lines
  and more files. Passing episodes edit by command ~8, failing ones ~12.

**Economy.** Wall-clock and turns are capped. `verify` is one call and costs no model
turn; never re-run a command that returned nothing; read with `sed -n 'A,Bp'` instead of
`cat`; `git diff --stat` instead of `git diff`.
