# bash tool notes

- State persists (same shell/container). Prefer compound commands: `cd /testbed && ...`.
- Every observation is capped; put what matters early (`| tail -40`, `| grep -E "FAILED|passed"`).
- The harness blocks exact-repeat read/test commands and replays the earlier output — treat the
  replay as if you had run it.
- Edits: `sed -i 's/old/new/' file`, or
  `python - <<'EOF' ... EOF` for multi-line replacements; verify with `sed -n 'A,Bp' file`.
- Add `timeout 110` in front of anything that might hang; async/event-loop tests can block.
- No network: `pip install`, `curl`, `git fetch` cannot work.
