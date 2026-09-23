#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime-env.sh"
export LD_LIBRARY_PATH="$BEV_ROOT/prism-build/bin:$BEV_ROOT/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="$BEV_ROOT/prism-build/bin/libbev-cuda-loader.so"
exec "$BEV_ROOT/prism-build/bin/llama-server" \
  --model "$BEV_ROOT/models/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
  --alias "$BEV_MODEL" --host 127.0.0.1 --port "$BEV_LLAMA_PORT" \
  --n-gpu-layers 99 --flash-attn on --ctx-size "${BEV_CTX_SIZE:-32768}" --parallel "${BEV_PARALLEL:-2}" \
  --batch-size "${BEV_BATCH_SIZE:-512}" --ubatch-size "${BEV_UBATCH_SIZE:-512}" \
  --threads "${BEV_THREADS:-8}" --threads-batch "${BEV_THREADS_BATCH:-8}" \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 1024 \
  --jinja --reasoning off \
  --no-context-shift --no-webui
