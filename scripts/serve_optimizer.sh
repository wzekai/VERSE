#!/usr/bin/env bash
# serve_optimizer.sh — the optimizer model, Qwen3.8-Flash-Next (FP8 checkpoint), on 4 GPUs
# (tensor parallel + expert parallel), port 8100.
#
# The optimizer runs in thinking mode with the checkpoint's default sampling: the code omits
# `temperature` for this model (verse/runtime/executor.py, _TEMP_DEPRECATED).
# The model needs a recent vLLM build that supports Qwen3.8.
#
# usage: PROP_GPU=0,1,2,3 bash scripts/serve_optimizer.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${PROP_GPU:-0,1,2,3}
# as in our runs: flashinfer kernels and the allreduce-RMS fusion pass off (they did not build
# with our CUDA toolchain); expert parallel is needed for the FP8 expert weights under TP4
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
exec ${VLLM:-vllm} serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --served-model-name qwen38-flash-next \
  --port "${PROP_PORT:-8100}" --host 127.0.0.1 \
  --tensor-parallel-size "${PROP_TP:-4}" --enable-expert-parallel \
  --moe-backend triton \
  --gpu-memory-utilization "${PROP_UTIL:-0.85}" \
  --max-model-len 262144 --max-num-batched-tokens "${PROP_BATCHED:-8192}" \
  --max-num-seqs 64 --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
