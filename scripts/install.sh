#!/usr/bin/env bash
# Install the pinned ternary runtime on a Linux x86_64 NVIDIA machine.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime-env.sh"

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  printf 'This installer requires Linux x86_64; see docs/INSTALL.md.\n' >&2
  exit 1
fi
for command in curl git cmake ninja c++ python3 sha256sum tar awk df stat ldd; do
  command -v "$command" >/dev/null || { printf 'Missing required command: %s\n' "$command" >&2; exit 1; }
done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or later is required")'
[ -f "$BEV_CODE/pyproject.toml" ] && [ -f "$BEV_CODE/patches/prism-selected-logprobs.patch" ] || {
  printf 'BEV_CODE must point to the complete Bev source checkout.\n' >&2; exit 1;
}
# Replacing shared libraries while a service maps them is unsafe.
for service in api llama; do
  pidfile="$BEV_ROOT/logs/$service.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      if bev_process_matches "$service" "$pid"; then
        printf 'Stop Bev with scripts/stop-services.sh before reinstalling its mapped libraries.\n' >&2
      else
        printf 'A live unexpected process is recorded in %s; inspect it before reinstalling.\n' "$pidfile" >&2
      fi
      exit 1
    fi
  fi
done
MODEL_REV=6ed5e12bf84b7a63069882c91dd9e9218647d17b
RUNTIME_TAG=prism-b10709-9a9394a
RUNTIME_REV=9a9394a895b96003ca842a6041cb28ac49a108f7
MODEL_FILE=Ternary-Bonsai-2-27B-PQ2_0.gguf
MODEL_SHA=3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1
RUNTIME_ARCHIVE_SHA=8aec67eb023b251712c7e6490f367b5671bf587eced1436a9b85f4a90c3b7d3d
CUDA_LIBRARY_SHA=a2f67b3e1a3fb476aba8b6fc1d2d2d7b25add3c4a169d77a679288492acc3762
mkdir -p "$BEV_ROOT"/{models,runtime,logs,artifacts}
if [ ! -s "$BEV_ROOT/models/$MODEL_FILE" ]; then
  available_kb=$(df -Pk "$BEV_ROOT" | awk 'NR==2 {print $4}')
  [ "$available_kb" -gt 11000000 ] || { printf 'Need at least 11 GB free for model and runtime.\n' >&2; exit 1; }
fi
if ! printf '%s  %s\n' "$MODEL_SHA" "$BEV_ROOT/models/$MODEL_FILE" | sha256sum --check --status 2>/dev/null; then
  if [ -f "$BEV_ROOT/models/$MODEL_FILE" ] && [ "$(stat -c %s "$BEV_ROOT/models/$MODEL_FILE")" -ge 7206168928 ]; then
    printf 'Existing model has unexpected bytes; choose a fresh BEV_ROOT.\n' >&2
    exit 1
  fi
  curl -fsSL --retry 5 -C - -o "$BEV_ROOT/models/$MODEL_FILE" \
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/$MODEL_REV/$MODEL_FILE"
fi
printf '%s  %s\n' "$MODEL_SHA" "$BEV_ROOT/models/$MODEL_FILE" | sha256sum --check
for notice in LICENSE NOTICE.txt; do
  curl -fsSL --retry 3 -o "$BEV_ROOT/models/$notice" \
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/$MODEL_REV/$notice"
done
archive="llama-$RUNTIME_TAG-bin-linux-cuda-12.8-x64.tar.gz"
if [ ! -x "$BEV_ROOT/runtime/llama-server" ]; then
  curl -fsSL --retry 3 -o "$BEV_ROOT/runtime/$archive" \
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/$RUNTIME_TAG/$archive"
  printf '%s  %s\n' "$RUNTIME_ARCHIVE_SHA" "$BEV_ROOT/runtime/$archive" | sha256sum --check
  tar -xzf "$BEV_ROOT/runtime/$archive" -C "$BEV_ROOT/runtime" --strip-components=1
fi
printf '%s  %s\n' "$CUDA_LIBRARY_SHA" "$BEV_ROOT/runtime/libggml-cuda.so.0.21.0" | sha256sum --check
dependencies=$(LD_LIBRARY_PATH="$BEV_ROOT/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$BEV_ROOT/runtime/libggml-cuda.so.0.21.0" 2>&1 || true)
if [[ "$dependencies" == *"not found"* ]]; then
  printf 'Pinned CUDA backend has unresolved libraries. Install CUDA 12.8 runtime/cuBLAS and a compatible NVIDIA driver; see docs/INSTALL.md.\n' >&2
  printf '%s\n' "$dependencies" >&2
  exit 1
fi
"$BEV_ROOT/runtime/llama-server" --version 2>&1 | grep -q 9a9394a || { printf 'Runtime version mismatch\n' >&2; exit 1; }
if [ ! -d "$BEV_ROOT/prism-source/.git" ]; then
  git clone --depth 1 --branch "$RUNTIME_TAG" https://github.com/PrismML-Eng/llama.cpp.git "$BEV_ROOT/prism-source"
fi
[ "$(git -C "$BEV_ROOT/prism-source" rev-parse HEAD)" = "$RUNTIME_REV" ] || { printf 'Runtime source revision mismatch\n' >&2; exit 1; }
if ! git -C "$BEV_ROOT/prism-source" apply --reverse --check "$BEV_CODE/patches/prism-selected-logprobs.patch" 2>/dev/null; then
  git -C "$BEV_ROOT/prism-source" apply --check "$BEV_CODE/patches/prism-selected-logprobs.patch"
  git -C "$BEV_ROOT/prism-source" apply "$BEV_CODE/patches/prism-selected-logprobs.patch"
fi
# Host source and prebuilt CUDA plugin must have the same pinned revision/ABI.
cmake -S "$BEV_ROOT/prism-source" -B "$BEV_ROOT/prism-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON \
  -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BEV_ROOT/prism-build" --target llama-server -j "${BEV_BUILD_JOBS:-6}"
c++ -O2 -shared -fPIC "$BEV_CODE/patches/bev-cuda-loader.cpp" \
  -L"$BEV_ROOT/runtime" -lggml-cuda -Wl,-rpath,"$BEV_ROOT/runtime" \
  -o "$BEV_ROOT/prism-build/bin/libbev-cuda-loader.so"
devices=$(GGML_BACKEND_PATH="$BEV_ROOT/prism-build/bin/libbev-cuda-loader.so" \
  LD_LIBRARY_PATH="$BEV_ROOT/prism-build/bin:$BEV_ROOT/runtime" \
  "$BEV_ROOT/prism-build/bin/llama-server" --list-devices 2>&1)
printf '%s\n' "$devices"
if ! grep -Eq 'CUDA[0-9]+:' <<<"$devices"; then
  printf 'No CUDA device was visible to the pinned runtime. Inspect the driver/runtime setup before starting Bev.\n' >&2
  exit 1
fi
python3 -m venv "$BEV_ROOT/.venv"
package="$BEV_CODE"
if [ "${BEV_INSTALL_DEV:-0}" = 1 ]; then package="$package[dev]"; fi
"$BEV_ROOT/.venv/bin/python" -m pip install -e "$package"
printf 'Installed Bev in %s. Run bash %s/scripts/start-services.sh with the same BEV_ROOT.\n' "$BEV_ROOT" "$BEV_CODE"
