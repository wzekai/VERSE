#!/bin/bash
# run_tb.sh — one Terminal-Bench harness-evolution run (or test evaluation) on a docker box.
#
# Same structure and env knobs as run_swe.sh. ROUNDS defaults to 3: every Terminal-Bench run
# in the paper used 3 rounds. Models are local vLLM servers listed in T2E_VLLM_MODELS; a model
# not listed there is called through the Bedrock Converse API.
#
# Required env:  CONFIG (e.g. tb/tb_verse_hx)  TAG
# Common env:    EXEC_MODEL  INT_MODEL (qwen38-flash-next)  ROUNDS (3)  CONC (24)
#                DO_TEST (1)  BLANK  EVAL_ONLY  EVAL_ROUND  TEST_ROUNDS   — as in run_swe.sh
# Box layout:    ROOT = repo root   CKPT = run outputs   TBROOT = checkouts of the task suites
#                BACKUP = optional rsync target for CKPT (empty = no backup)
#
# Example:  CONFIG=tb/tb_harnessx TAG=-r1 bash scripts/run_tb.sh
set -uxo pipefail
: "${CONFIG:?}"; : "${TAG:?}"
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
CKPT=${CKPT:-$ROOT/out}
TBROOT=${TBROOT:-$ROOT/tb_tasks}
BACKUP=${BACKUP:-}                              # optional rsync target for CKPT
ROUNDS=${ROUNDS:-3}; CONC=${CONC:-24}; DO_TEST=${DO_TEST:-1}; BLANK=${BLANK:-}
EVAL_ONLY=${EVAL_ONLY:-}; EVAL_ROUND=${EVAL_ROUND:-}
TRAIN_TASKS=${TRAIN_TASKS:-104}; VAL_TASKS=${VAL_TASKS:-50}
EXEC_MODEL=${EXEC_MODEL:-qwen38-27b}
INT_MODEL=${INT_MODEL:-qwen38-flash-next}
EXEC_REGIONS=${EXEC_REGIONS:-us-west-2}        # ignored by the vLLM transport; kept for the CLI
export T2E_VLLM_MODELS=${T2E_VLLM_MODELS:-"qwen38-flash-next=http://127.0.0.1:8100,qwen38-27b=http://127.0.0.1:8101"}
export TB_TRAIN=${TB_TRAIN:-tb_train_v3.parquet}; export TB_VAL=${TB_VAL:-tb_val_v4.parquet}
export TB_TEST=${TB_TEST:-tb_test.parquet}
# GOLD_INVALID: basename under data/ of a gold-scan invalid list for TB_TEST; empty = every
# task judged (the sealed tb_test pool needs none)
GOLD_INVALID=${GOLD_INVALID:-}
ARM=$(basename "$CONFIG")
mkdir -p "$CKPT/logs" "$CKPT/tmp"
LOG=$CKPT/logs/tb_${ARM}${TAG}_$(date +%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
PY=${PY:-python}

cd "$ROOT"
export PYTHONPATH=$ROOT
export TMPDIR=$CKPT/tmp

# --- preflight: data, task checkouts, both LLM servers, docker ---
for f in "$TB_TRAIN" "$TB_VAL" "$TB_TEST"; do
  [ -f "data/$f" ] || { echo "MISSING data/$f"; exit 1; }
done
[ -d "$TBROOT/original-tasks" ] || { echo "MISSING $TBROOT/original-tasks (Terminal-Bench 1.x checkout, see data/README.md)"; exit 1; }
for m in ${T2E_VLLM_MODELS//,/ }; do
  urls=${m#*=}   # one URL, or several replicas "url1|url2|..." (replica affinity in the transport)
  for u in ${urls//|/ }; do
    curl -sf -m 10 "$u/v1/models" >/dev/null || { echo "LLM_DOWN $m ($u)"; exit 1; }
  done
done
docker info | grep "Server Version" || { echo DOCKER_DEAD; exit 1; }

sync_ckpt() { [ -n "$BACKUP" ] && rsync -a --exclude tmp "$CKPT/" "$BACKUP/" >/dev/null 2>&1 || true; }
( while true; do sleep 300; sync_ckpt; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; sync_ckpt' EXIT      # no orphaned sync loop per run

# --- TB task definitions: labeled checkouts (rows carry tb_src=tb1|tb2|tblite|tbpro|deepswe) ---
export T2T_TB_TASKS="tb1=$TBROOT/original-tasks:tb2=$TBROOT/terminal-bench-2:tblite=$TBROOT/OpenThoughts-TBLite:tbpro=$TBROOT/terminal-bench-pro:deepswe=$TBROOT/deepswe:tb21=$TBROOT/terminal-bench-2-1:tb4=$TBROOT/terminal-bench-4"

# --- TB images: local docker cache only (T2E_TB_ECR_REPO unset: no registry pull or push).
# Build every task image once up front (a cached image costs one `docker image inspect`).
export T2E_SWE_DOCKER=docker
export T2E_TB_BUILD_TIMEOUT=${T2E_TB_BUILD_TIMEOUT:-2400}
if [ "${TB_PREBUILD:-1}" = "1" ]; then
ROOT=$ROOT $PY - <<'PYBUILD'
import json, os, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("T2E_SWE_DOCKER", "docker")
import sys; sys.path.insert(0, os.environ["ROOT"])
from verse.runtime.tb_env import TBContainerSession
tasks = set()
for f in (os.environ.get("TB_TRAIN", "tb_train_v3.parquet"),
          os.environ.get("TB_VAL", "tb_val_v4.parquet"),
          os.environ.get("TB_TEST", "tb_test.parquet")):
    p = f"data/{f}"
    if not os.path.exists(p): continue
    df = pd.read_parquet(p)
    for _, r in df.iterrows():
        ei = r["extra_info"]; d = ei if isinstance(ei, dict) else json.loads(ei)
        ti = d.get("tb_instance", d)
        inst = json.loads(ti) if isinstance(ti, str) else ti
        tid = inst.get("instance_id")
        if tid: tasks.add((tid, inst.get("tb_src", "")))
t0 = time.time()
print(f"TB_PREBUILD_START {len(tasks)} tasks", flush=True)
def build(t):
    tid, src = t
    ok = TBContainerSession(tid, src=src).prebuild()
    if not ok: print(f"TB_PREBUILD_FAIL {tid}", flush=True)
    return ok
with ThreadPoolExecutor(max_workers=12) as ex:
    ok = sum(ex.map(build, sorted(tasks)))
print(f"TB_PREBUILD_DONE {ok}/{len(tasks)} in {time.time()-t0:.0f}s", flush=True)
PYBUILD
fi
# Test-suite timeout. At 1800 s the test suites of build-initramfs-qemu and solve-maze-challenge
# (over 30 minutes) always time out and score NaN, so the effective training pool is 101 of 104
# tasks for every method. Keep it fixed across runs that are compared.
export T2E_TB_TEST_TIMEOUT=${T2E_TB_TEST_TIMEOUT:-1800}

export T2E_CERT_CACHE_DIR=$CKPT/cert_cache/$ARM$TAG
export T2E_EPISODE_WALL_S=${T2E_EPISODE_WALL_S:-2700}
export T2E_SWE_CONTAINER_CPUS=${T2E_SWE_CONTAINER_CPUS:-2}

OUT=$CKPT/$ARM$TAG; mkdir -p "$OUT"
CFG=$ROOT/verse/configs/$CONFIG.yaml
if [ -n "$BLANK" ]; then
  RC=0
  echo "BLANK_MODE: skipping driver"
elif [ -n "$EVAL_ONLY" ]; then
  RC=0
  echo "EVAL_ONLY: skipping driver, scoring the ckpt's val-picked round"
else
  $PY -m verse.evolution.driver --config "$CFG" --out "$OUT" \
    --set data.train_parquet=data/$TB_TRAIN \
    --set data.val_parquet=data/$TB_VAL \
    --set data.train_tasks=$TRAIN_TASKS --set data.val_tasks=$VAL_TASKS \
    --set rounds=$ROUNDS --set sweep.concurrency=$CONC \
    --set executor.model=$EXEC_MODEL --set executor.regions=$EXEC_REGIONS \
    --set intervener.model=$INT_MODEL \
    ${INT_REGIONS:+--set intervener.regions=$INT_REGIONS}
  RC=$?
  echo "DRIVER_EXIT $RC"
fi

if [ $RC -eq 0 ] && [ "$DO_TEST" = "1" ]; then
  kill $SYNC_PID 2>/dev/null || true
  sync_ckpt
  GI=""
  [ -n "$GOLD_INVALID" ] && [ -f "$ROOT/data/$GOLD_INVALID" ] && GI=$ROOT/data/$GOLD_INVALID
  BLANK_FLAG=""
  [ -n "$BLANK" ] && BLANK_FLAG="--blank --base-prompt TB"
  [ -n "$EVAL_ROUND" ] && BLANK_FLAG="--round $EVAL_ROUND"
  # eval 1: the val-selected round (or the initial TB harness)
  $PY -m verse.evolution.eval --arm-dir "$OUT" \
    --test-parquet data/$TB_TEST --conc $CONC --repeats 3 \
    --model "$EXEC_MODEL" --regions "$EXEC_REGIONS" \
    ${GI:+--invalid-tasks "$GI"} $BLANK_FLAG
  RC=$?
  echo "EVAL_EXIT $RC"
  # eval 2: the last round
  if [ -z "$BLANK_FLAG" ] && [ -z "$EVAL_ONLY" ] && [ "$ROUNDS" -gt 0 ]; then
    sync_ckpt
    $PY -m verse.evolution.eval --arm-dir "$OUT" \
      --test-parquet data/$TB_TEST --conc $CONC --repeats 3 --round -1 \
      --model "$EXEC_MODEL" --regions "$EXEC_REGIONS" \
      ${GI:+--invalid-tasks "$GI"}
    echo "EVAL_LASTROUND_EXIT $?"
  fi
  for RN in ${TEST_ROUNDS:-}; do
    sync_ckpt
    $PY -m verse.evolution.eval --arm-dir "$OUT" \
      --test-parquet data/$TB_TEST --conc $CONC --repeats 3 --round $RN \
      --model "$EXEC_MODEL" --regions "$EXEC_REGIONS" \
      ${GI:+--invalid-tasks "$GI"}
    echo "EVAL_ROUND${RN}_EXIT $?"
  done
fi
sync_ckpt
echo "RUN_DONE $ARM$TAG rc=$RC"
exit $RC
