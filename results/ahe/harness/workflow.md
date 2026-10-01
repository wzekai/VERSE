# Workflow (this order is the point - measured, do not reorder)

**Phase 0 - read the issue as a contract (0 commands).**
From the issue text alone: the file/subsystem it names and the literal names it promises
(functions, arguments, defaults, exception types, CLI flags, message wording). The grading
tests are written against *those* names; they are hidden, so the issue text is your only
spec. The harness pins this list as a CONTRACT CARD.

**Phase 1 - get a check that FAILS, before you edit (1-4 commands).**
Call **`spec`**: one model call that reads the ISSUE TEXT ONLY (no repo, no diff), writes
`/tmp/spec_test.py` and runs it. It should be RED - that failure tells you what the un-fixed
code lacks, and it comes from a reader that never saw your implementation. Your target is
`cd /testbed && python -m pytest /tmp/spec_test.py -q`: RED now, GREEN at the end. You may
rewrite it against the report's own example as `/tmp/contract_test.py`.
Why this phase exists (measured): 0 of 110 episodes wrote a check before their first edit, so
every check they ran afterwards only asserted what their own patch already did; and a green
local suite predicts nothing - the last verification was green in 61/75 FAILED and 30/35
PASSED episodes, because the repo's tests were green before your patch too.

**Phase 2 - locate (<= 10 commands) and EDIT small (by ~command 10).**
`grep -rn "<symbol>" /testbed --include=*.py | head -30`, `sed -n 'A,Bp' file`. Read the
closest existing test file for the symbol you edit: the hidden test is an *updated version of
that file*, so its parametrisation shows the dimensions the new test sweeps. Never re-run a
command that returned nothing; keep the patch to the file the report names (passing episodes
median 1 file / +4 added lines; first edit at command ~8, failing ones ~12).
```bash
cd /testbed && python3 - <<'EOF'
p = 'path/to/file.py'
s = open(p).read()
open(p, 'w').write(s.replace(OLD, NEW))
EOF
cd /testbed && git diff --stat
```
An episode whose diff is empty grades 0 - a plausible patch always beats a perfect analysis.

**Phase 3 - `verify` (one call), then drive the differential to PROVEN.**
`verify` is deterministic (no model writes it): it re-runs `/tmp/spec_test.py` and
`/tmp/contract_test.py`, shows the patch radar, compiles your files, runs the tests that
cover the change. Its first line is the verdict:
`PROVEN` (a check red at HEAD is green now) / `UNPROVEN` (nothing ever failed at HEAD - your
patch is unverified) / `NOT YET` or `BROKEN` (a check is still red - your patch does not
deliver the report's own contract; do not submit).

**Phase 4 - submit.**
Check the patch against every name in the CONTRACT CARD - spelling, module, argument order,
defaults, exception type, and the invalid/edge case the report shows. That is all the hidden
test can fail on. Then `submit`; do not keep exploring.
