#!/bin/bash
# build_swe_ood_images.sh — build the task images of the OOD test set (SWE-rebench July 2026
# release, Harbor format) from each task's environment/Dockerfile
# (FROM docker.io/swerebenchv2/<id>:v0.1.0 + uv + /logs), tag them as recorded in the manifest
# (sweb07/<id>:v0.1.0), and check that the repository checkout exists. Docker Hub limits
# anonymous pulls, so after PULL_BATCH builds the script waits for the next window.
# Idempotent (skips existing tags).
#
# usage: DS=<dataset dir> bash scripts/build_swe_ood_images.sh [data/swe_test_0726_manifest.json]
set -u
: "${DS:?set DS to the downloaded dataset dir (.../swe-rebench-07-2026)}"
MAN=${1:-data/swe_test_0726_manifest.json}
LOG=${LOG:-build_swe_ood_images.log}; PULL_BATCH=${PULL_BATCH:-90}; WINDOW_S=${WINDOW_S:-3700}
TASKS=$(mktemp)
python3 - "$MAN" <<'PY' > "$TASKS"
import json,sys
for t in json.load(open(sys.argv[1]))["tasks"]: print(t["task_dir"], t["docker_image"], t["repo_dir"], t["hub_image"])
PY
n=0; built=0; fail=0; window_start=$(date +%s)
while read -r task img repo hub; do
  if docker image inspect "$img" >/dev/null 2>&1; then echo "$(date -u +%FT%TZ) skip $img (exists)" >> "$LOG"; continue; fi
  if [ $n -ge $PULL_BATCH ]; then
    wait_s=$(( WINDOW_S - ($(date +%s) - window_start) )); if [ $wait_s -gt 0 ]; then echo "$(date -u +%FT%TZ) pull window: sleeping $wait_s s" >> "$LOG"; sleep $wait_s; fi
    n=0; window_start=$(date +%s)
  fi
  n=$((n+1))
  echo "$(date -u +%FT%TZ) build $img <- $hub" >> "$LOG"
  if docker build -q -t "$img" "$DS/$task/environment" >> "$LOG" 2>&1; then
    if docker run --rm --network none "$img" bash -c "test -d $repo/.git && echo REPO_OK || { ls -d /*/.git; echo REPO_MISSING; }" >> "$LOG" 2>&1; then built=$((built+1)); else echo "$(date -u +%FT%TZ) REPO_CHECK_FAIL $img" >> "$LOG"; fi
  else echo "$(date -u +%FT%TZ) BUILD_FAIL $img" >> "$LOG"; fail=$((fail+1)); fi
done < "$TASKS"
rm -f "$TASKS"
echo "$(date -u +%FT%TZ) BUILD_DONE built=$built fail=$fail total=$(docker images --format '{{.Repository}}' | grep -c '^sweb07/')" >> "$LOG"
