# Loop protocol (follow it literally)

Turns are limited (and the episode also ends the moment you reply without a tool call), so the
budget is: locate ~10% / edit ~20% / verify+iterate ~60% / review+submit ~10%.

1. **Locate** (max ~6-8 commands): `grep -rn` for the symbol, message or behaviour named in the
   issue; `sed -n` the match and its callers; read the matching test for the exact contract.
2. **Edit early** (turn 10 at the latest): apply the change to the SOURCE in the same command you
   decided it — `python - <<'PY' ... open(p,'w').write(new) PY`, a heredoc, or `sed -i`. Minimal,
   focused edits that implement what the ISSUE literally says.
3. **Verify** with the repo interpreter, on the WHOLE covering test file (a neighbouring test you
   broke costs you as much as a test you never fixed):
   `/opt/conda/envs/testbed/bin/python -m pytest -q <file>` — drop `-x` when checking for
   regressions — plus a 5-line reproduction of the issue's example under `/tmp/repro.py`.
   Re-verify after EVERY edit.
4. **Review**: `git --no-pager diff` — it must contain the source change, no debug prints, no edits
   under tests/.
5. **Submit** after a test run confirms the behaviour. Two or three verification runs are enough;
   never re-read a file you already read, never re-run an identical command.
