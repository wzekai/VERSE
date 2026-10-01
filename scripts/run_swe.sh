#!/bin/bash
# run_swe.sh — one SWE-rebench harness-evolution run (or test evaluation) on a docker box.
#
# Runs the driver (6 rounds of evolution), then scores the val-selected round and the last
# round on the test set (3 repeats each). Models are local vLLM servers (scripts/serve_*.sh)
# listed in T2E_VLLM_MODELS; a model not listed there is called through the Bedrock Converse API.
#
# Required env:  CONFIG (e.g. self_teacher/verse_mh)  TAG (run-name suffix, e.g. -r1)
# Common env:    EXEC_MODEL (qwen38-27b)  INT_MODEL (qwen38-flash-next)  ROUNDS (6)  CONC (24)
#                DO_TEST (1)   BLANK=1: only score the initial harness h0
#                EVAL_ONLY=1: skip evolution, score the val-selected round of an existing run
#                EVAL_ROUND=N / -1: score round N / the last round   TEST_ROUNDS="0 1": extra rounds
#                EVAL_WORKSPACE=<dir>: skip evolution, score that harness directory as-is
#                (e.g. results/verse_mh/harness)
#                EVAL_CONC (CONC)  EVAL_PARALLEL=1: score val-selected and last round concurrently
#                SWE_TRAIN / SWE_VAL / SWE_TEST / GOLD_INVALID: file names under data/
# Box layout:    ROOT = repo root (verse/ + data/)   CKPT = run outputs
#                BACKUP = optional rsync target for CKPT (empty = no backup)
#
# Example:  CONFIG=self_teacher/verse_mh TAG=-r1 bash scripts/run_swe.sh
set -uxo pipefail
: "${CONFIG:?}"; : "${TAG:?}"
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CKPT=${CKPT:-$ROOT/out}
BACKUP=${BACKUP:-}                              # optional rsync target for CKPT
ROUNDS=${ROUNDS:-6}; CONC=${CONC:-24}; DO_TEST=${DO_TEST:-1}; BLANK=${BLANK:-}
EVAL_ONLY=${EVAL_ONLY:-}; EVAL_ROUND=${EVAL_ROUND:-}; EVAL_WORKSPACE=${EVAL_WORKSPACE:-}
[ -n "$EVAL_WORKSPACE" ] && EVAL_WORKSPACE=$(cd "$EVAL_WORKSPACE" && pwd)
# EVAL_CONC: --conc for the test evals (default CONC). EVAL_PARALLEL=1: run the evals of the
# val-selected round and the last round concurrently when they differ (each at EVAL_CONC).
EVAL_CONC=${EVAL_CONC:-$CONC}; EVAL_PARALLEL=${EVAL_PARALLEL:-}
TRAIN_TASKS=${TRAIN_TASKS:-110}; VAL_TASKS=${VAL_TASKS:-50}
export SWE_TEST=${SWE_TEST:-swe_test.parquet}
# GOLD_INVALID: basename under data/ of the gold-scan invalid list matching SWE_TEST
GOLD_INVALID=${GOLD_INVALID:-gold_invalid_test.json}
EXEC_MODEL=${EXEC_MODEL:-qwen38-27b}
INT_MODEL=${INT_MODEL:-qwen38-flash-next}
EXEC_REGIONS=${EXEC_REGIONS:-us-west-2}        # ignored by the vLLM transport; kept for the CLI
# served name = base URL; several replicas of one model: "name=url1|url2|..."
export T2E_VLLM_MODELS=${T2E_VLLM_MODELS:-"qwen38-flash-next=http://127.0.0.1:8100,qwen38-27b=http://127.0.0.1:8101"}
export SWE_TRAIN=${SWE_TRAIN:-swe_train_v6.parquet}; export SWE_VAL=${SWE_VAL:-swe_val_v6.parquet}
ARM=$(basename "$CONFIG")
mkdir -p "$CKPT/logs" "$CKPT/tmp"
LOG=$CKPT/logs/swe_${ARM}${TAG}_$(date +%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
PY=${PY:-python}

cd "$ROOT"
export PYTHONPATH=$ROOT
export TMPDIR=$CKPT/tmp

# --- preflight: data, the vLLM servers, docker ---
NEED=("$SWE_TEST")   # train/val data are read only by the driver
[ -z "$BLANK$EVAL_ONLY$EVAL_WORKSPACE" ] && NEED+=("$SWE_TRAIN" "$SWE_VAL")
for f in "${NEED[@]}"; do
  [ -f "data/$f" ] || { echo "MISSING data/$f"; exit 1; }
done
for m in ${T2E_VLLM_MODELS//,/ }; do
  urls=${m#*=}   # one URL, or several replicas "url1|url2|..." (replica affinity in the transport)
  for u in ${urls//|/ }; do
    curl -sf -m 10 "$u/v1/models" >/dev/null || { echo "LLM_DOWN $m ($u)"; exit 1; }
  done
done
docker info | grep "Server Version" || { echo DOCKER_DEAD; exit 1; }

# --- optional checkpoint backup (rsync to $BACKUP every 5 minutes) ---
sync_ckpt() { [ -n "$BACKUP" ] && rsync -a --exclude tmp "$CKPT/" "$BACKUP/" >/dev/null 2>&1 || true; }
( while true; do sleep 300; sync_ckpt; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_ckpt' EXIT      # no orphaned sync loop per run

# --- runtime env (the settings of the paper's runs) ---
# T2E_SWE_ECR_PREFIX: optional registry mirror for the task images; empty = the public names
# stored in the task rows (swerebench/... on Docker Hub, or the locally built OOD images)
export T2E_SWE_ECR_PREFIX=${T2E_SWE_ECR_PREFIX:-}
export T2E_SWE_EXEC_BACKEND=docker
export T2E_SWE_DOCKER=docker
export T2E_SWE_DOCKER_RUN_ARGS="--sysctl net.ipv6.conf.all.disable_ipv6=0"
export T2E_SWE_SCRUB_GIT=1
export T2E_CERT_CACHE_DIR=$CKPT/cert_cache/$ARM$TAG
export T2E_EPISODE_WALL_S=${T2E_EPISODE_WALL_S:-2700}
export T2E_SWE_EXEC_TIMEOUT=${T2E_SWE_EXEC_TIMEOUT:-120}
export T2E_SWE_CONTAINER_CPUS=${T2E_SWE_CONTAINER_CPUS:-2}
export T2E_SWE_START_TIMEOUT=${T2E_SWE_START_TIMEOUT:-1200}

# the output dir carries the TAG, so one box can host several runs
OUT=$CKPT/$ARM$TAG; mkdir -p "$OUT"
CFG=$ROOT/verse/configs/$CONFIG.yaml
if [ -n "$BLANK" ]; then
  RC=0
  echo "BLANK_MODE: skipping driver"
elif [ -n "$EVAL_WORKSPACE" ]; then
  RC=0
  echo "EVAL_WORKSPACE: skipping driver, scoring $EVAL_WORKSPACE"
elif [ -n "$EVAL_ONLY" ]; then
  RC=0
  echo "EVAL_ONLY: skipping driver, scoring the ckpt's val-picked round"
else
  $PY -m verse.evolution.driver --config "$CFG" --out "$OUT" \
    --set data.train_parquet=data/$SWE_TRAIN \
    --set data.val_parquet=data/$SWE_VAL \
    --set data.train_tasks=$TRAIN_TASKS --set data.val_tasks=$VAL_TASKS \
    --set rounds=$ROUNDS --set sweep.concurrency=$CONC \
    --set executor.model=$EXEC_MODEL --set executor.regions=$EXEC_REGIONS \
    --set intervener.model=$INT_MODEL \
    ${INT_REGIONS:+--set intervener.regions=$INT_REGIONS} \
    ${PROBE_BUDGET:+--set probes.budget=$PROBE_BUDGET}
  RC=$?
  echo "DRIVER_EXIT $RC"
fi

if [ $RC -eq 0 ] && [ "$DO_TEST" = "1" ]; then
  # stop the 5-minute background sync during the test evals (its IO/CPU load adds noise);
  # sync between evals instead
  kill $SYNC_PID 2>/dev/null || true
  sync_ckpt
  GI=""
  [ -n "$GOLD_INVALID" ] && [ -f "$ROOT/data/$GOLD_INVALID" ] && GI=$ROOT/data/$GOLD_INVALID
  BLANK_FLAG=""
  [ -n "$BLANK" ] && BLANK_FLAG="--blank"
  [ -n "$EVAL_ROUND" ] && BLANK_FLAG="--round $EVAL_ROUND"
  [ -n "$EVAL_WORKSPACE" ] && BLANK_FLAG="--workspace $EVAL_WORKSPACE"
  run_eval() {   # $1 = extra eval flags: "" (val pick) | "--blank" | "--round N" | "--round -1"
    $PY -m verse.evolution.eval --arm-dir "$OUT" \
      --test-parquet data/$SWE_TEST --conc $EVAL_CONC --repeats 3 \
      --model "$EXEC_MODEL" --regions "$EXEC_REGIONS" \
      ${GI:+--invalid-tasks "$GI"} $1
  }
  pick_is_last() {   # selected round is the last round, so eval 2 would repeat eval 1
    $PY - "$OUT/rounds.jsonl" <<'PYEOF'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
fin = [r for r in rs if r.get("round") == "final"]
nums = [r["round"] for r in rs if isinstance(r.get("round"), int)]
sys.exit(0 if fin and nums and fin[-1].get("best_round") == max(nums) else 1)
PYEOF
  }
  WANT_LAST=""
  [ -z "$BLANK_FLAG" ] && [ -z "$EVAL_ONLY" ] && [ "$ROUNDS" -gt 0 ] && WANT_LAST=1
  if [ -n "$WANT_LAST" ] && [ -n "$EVAL_PARALLEL" ] && ! pick_is_last; then
    # EVAL_PARALLEL=1: eval 1 (val-selected round) and eval 2 (last round) run concurrently on
    # different harnesses with separate output files (test110.json / test110_r<N>.json,
    # traj_test_rep* / traj_test_r<N>_rep*). The executor then serves 2 x EVAL_CONC episodes.
    run_eval "" & P1=$!
    run_eval "--round -1" & P2=$!
    wait $P1; RC=$?
    echo "EVAL_EXIT $RC"
    wait $P2; echo "EVAL_LASTROUND_EXIT $?"
  else
    # eval 1: the val-selected round (or the initial harness, or EVAL_ROUND's harness)
    run_eval "$BLANK_FLAG"; RC=$?
    echo "EVAL_EXIT $RC"
    # eval 2: the last round, to compare validation-based selection with taking the last round
    if [ -n "$WANT_LAST" ]; then
      sync_ckpt
      run_eval "--round -1"; echo "EVAL_LASTROUND_EXIT $?"
    fi
  fi
  # extra per-round evals: TEST_ROUNDS="0 1" scores each listed round's harness
  for RN in ${TEST_ROUNDS:-}; do
    sync_ckpt
    run_eval "--round $RN"; echo "EVAL_ROUND${RN}_EXIT $?"
  done
fi
sync_ckpt
echo "RUN_DONE $ARM$TAG rc=$RC"
exit $RC
