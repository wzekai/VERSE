#!/bin/bash
# add_python_to_swe_ood_images.sh — add a uv-managed Python 3.12 to every OOD task image that has
# no python3. The Harbor verifier runs `uv run parser.py` when the image has no python3, which
# would download an interpreter at grading time; installing it once keeps grading fast and
# repeatable. Re-tags the same image name. Run after build_swe_ood_images.sh.
set -u; LOG=${LOG:-build_swe_ood_images.log}; n=0; fail=0
for img in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^sweb07/'); do
  if docker run --rm --network none "$img" bash -c 'command -v python3 >/dev/null || ls /root/.local/share/uv/python 2>/dev/null | grep -q cpython' 2>/dev/null; then n=$((n+1)); continue; fi
  printf 'FROM %s\nRUN uv python install 3.12 || /usr/local/bin/uv python install 3.12\n' "$img" | docker build -q -t "$img" - >> "$LOG" 2>&1 && n=$((n+1)) || { echo "$(date -u +%FT%TZ) PY_LAYER_FAIL $img" >> "$LOG"; fail=$((fail+1)); }
done
echo "$(date -u +%FT%TZ) PYTHON_LAYER_DONE ok=$n fail=$fail" >> "$LOG"
