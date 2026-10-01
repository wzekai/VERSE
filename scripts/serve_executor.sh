#!/usr/bin/env bash
# serve_executor.sh — one vLLM replica of the frozen executor, Qwen3.8-27B (BF16).
#
# The paper's runs used four independent single-GPU replicas (ports 8101-8104) and listed all of
# them in T2E_VLLM_MODELS ("qwen38-27b=http://127.0.0.1:8101|http://127.0.0.1:8102|..."); the
# transport pins every episode to one replica so its prefix cache stays warm. Each replica ran
# with --max-num-seqs 44 (EXEC_SEQS=44). Thinking is on with reasoning_effort=low, so the thinking
# fits the executor's 4096-token turn budget; temperature 0 is set by the code, not here.
# The model needs a recent vLLM build that supports Qwen3.8.
#
# usage: EXEC_GPU=0 EXEC_PORT=8101 EXEC_SEQS=44 bash scripts/serve_executor.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${EXEC_GPU:-0}
# as in our runs: flashinfer kernels off (they did not build with our CUDA toolchain)
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
exec ${VLLM:-vllm} serve "${EXEC_MODEL:-Qwen/Qwen3.8-27B}" \
  --served-model-name "${EXEC_NAME:-qwen38-27b}" \
  --port "${EXEC_PORT:-8101}" --host 127.0.0.1 \
  --data-parallel-size "${EXEC_DP:-1}" --data-parallel-size-local "${EXEC_DP:-1}" \
  --gpu-memory-utilization "${EXEC_UTIL:-0.85}" \
  --max-model-len 262144 \
  --max-num-seqs "${EXEC_SEQS:-44}" --enable-prefix-caching \
  --max-num-batched-tokens "${EXEC_BATCHED:-16384}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs "{\"enable_thinking\": ${EXEC_THINK:-true}, \"reasoning_effort\": \"${EXEC_EFFORT:-low}\"}"
