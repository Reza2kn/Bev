# Installation

Bev has a small Python API and a separate native model server. Install both from
a clone or source release of this repository. The Python wheel contains the API;
it does not bundle the model, CUDA libraries, or the native server.

## Supported installation path

- **Linux x86_64 with an NVIDIA GPU.** The validated machine used an RTX 5080
  Laptop GPU with 16,303 MiB VRAM and NVIDIA driver 580.178.04. Other NVIDIA cards
  need their own runtime and memory verification; no minimum-VRAM claim is made.
- **CUDA 12.8 runtime and cuBLAS libraries**, visible to the dynamic linker:
  `libcudart.so.12`, `libcublas.so.12`, and its dependency `libcublasLt.so.12`.
  The NVIDIA driver supplies `libcuda.so.1`. Installing CUDA 13 alone does not
  supply these CUDA 12 library names. The installer checks unresolved libraries.
- A driver compatible with both the GPU and CUDA 12.8. NVIDIA lists Linux driver
  570.26 for CUDA 12.8 GA; a particular GPU can require a newer driver. See the
  [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).
- The pinned CUDA plugin requires **glibc 2.34 or newer** and a `libstdc++`
  providing **GLIBCXX_3.4.30**. Those are binary requirements, not a claim that all
  distributions meeting them have been tested.
- **Python 3.11+**, its `venv` module, a C/C++ compiler, CMake 3.14+, Ninja, Git,
  curl, OpenSSL development headers, and standard GNU/Linux command-line tools.
- At least **11 GB free disk space** for a fresh model/runtime installation, with
  additional space for build outputs, environments, and evaluation artifacts.

For example, on Ubuntu 24.04 the host build tools can be installed with:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build git curl python3 python3-venv libssl-dev
```

Install the NVIDIA driver and CUDA 12.8 runtime/cuBLAS for your distribution
separately. Bev does not change system drivers or system CUDA installations, and
does not require `nvcc`: it rebuilds host code while using pinned precompiled GPU
kernels.

macOS, Windows, WSL, ARM, AMD GPUs, CPU-only execution, and other quantization
formats are outside this installer’s validated scope. The fact that upstream
llama.cpp supports a platform does not establish that this pinned PQ2_0 runtime
and loader have been verified there.

## Install and start

From the repository root:

```bash
export BEV_ROOT="$HOME/.local/share/bev"
bash scripts/install.sh
bash scripts/start-services.sh
curl --fail http://127.0.0.1:18781/health
```

The installer downloads the original Prism model and runtime at pinned revisions,
verifies their SHA-256 hashes, applies the selected-token scoring patch, builds
the native server and CUDA loader, checks CUDA device visibility, and installs the
API in a virtual environment. It does not start inference services automatically.

The model is `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7,206,168,928 bytes), from Prism
revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`. No weights are retrained or
modified. A full existing file with the wrong hash is refused rather than silently
reused. Incomplete downloads can resume.

`BEV_ROOT` defaults to `$XDG_DATA_HOME/bev` when `XDG_DATA_HOME` is set, otherwise
`$HOME/.local/share/bev`. `BEV_CODE` defaults to the source checkout containing the
scripts. Keep that checkout available because the Python install is editable.
The old `BEV_HOME` variable remains an alias for `BEV_ROOT`; conflicting values
are rejected. Existing installations can continue by setting their original
`BEV_HOME` explicitly. `install-stallion.sh` remains a compatibility wrapper.

Both servers bind to loopback. Native inference uses port 18780 and the decision
API uses 18781. Health reports the actual model, per-slot context, selected-token
capability, and source hashes captured at process startup. A successful health
response establishes readiness; a functional call is still needed to verify
inference on a new machine.

```bash
"$BEV_ROOT/.venv/bin/python" scripts/smoke_api.py --url http://127.0.0.1:18781
```

The smoke script runs GPU inference and writes a functional-check artifact. It
is not an accuracy benchmark. The more detailed runtime parity probe is:

```bash
"$BEV_ROOT/.venv/bin/python" scripts/probe_runtime.py \
  --url http://127.0.0.1:18780 --output "$BEV_ROOT/artifacts/runtime-probe.json"
```

## Configuration

Export settings before starting the scripts, and keep the same `BEV_ROOT` for
install, start and stop. Settings are not silently written to shell profiles.

| Variable | Default | Meaning |
|---|---|---|
| `BEV_ROOT` | User data directory described above | Models, runtime, build, virtual environment, logs |
| `BEV_CODE` | This source checkout | API and patch source |
| `BEV_LLAMA_PORT` | `18780` | Loopback native-server port |
| `BEV_PORT` | `18781` | Loopback decision-API port |
| `BEV_MODEL` | `bev-bonsai-27b` | Served alias and expected API model identity |
| `BEV_CTX_SIZE` | `32768` | Native total context allocation |
| `BEV_PARALLEL` | `2` | Native concurrent slots |
| `BEV_BATCH_SIZE`, `BEV_UBATCH_SIZE` | `512`, `512` | Native prefill batch settings |
| `BEV_THREADS`, `BEV_THREADS_BATCH` | `8`, `8` | Native CPU thread settings |
| `BEV_BUILD_JOBS` | `6` | Concurrent host compilation jobs |
| `BEV_INSTALL_DEV` | `0` | Set `1` to install the test extras |
| `BEV_START_TIMEOUT_SECONDS` | `180` | Readiness polling budget, plus bounded HTTP waits |
| `BEV_HTTP_TIMEOUT` | `300` | API-to-native-server timeout in seconds |
| `BEV_INITIAL_N_PROBS` | `512` | Initial top-N size when the selected-token patch is absent |

The measured default configuration has **16,384 tokens per slot**, including the
single answer token. Consult `/health.max_model_len` for the actual loaded limit.
Changing context/parallelism affects memory and latency. Requests exceeding the
loaded limit are rejected, not truncated. The scripts retain Q8_0 KV cache,
disabled thinking, full GPU offload, and no context shifting.

For a backend managed separately, `start-api.sh` also accepts `BEV_LLAMA_URL` and
`BEV_LLAMA_API_KEY`. `start-services.sh` manages only its local native server.
The decision API itself has no authentication middleware; use an authenticated
gateway if exposing it beyond loopback.

## Stop, reinstall and inspect

```bash
bash scripts/stop-services.sh
bash scripts/install.sh
```

Logs are in `$BEV_ROOT/logs/api.log` and `$BEV_ROOT/logs/llama-patched.log`.
The service scripts track and check the exact configured executables before
stopping them. They refuse an unexpected live PID or an already healthy service
owned by another installation. The installer refuses to overwrite libraries
tracked as in use by this installation; stop foreground/manual servers yourself
before rebuilding their libraries.

The GPU plugin and rebuilt host code must stay on the same pinned Prism revision
`9a9394a895b96003ca842a6041cb28ac49a108f7`. Do not mix arbitrary newer runtimes with
the loader or patch. The original upstream model and runtime licenses remain
applicable; see the repository’s third-party notices.

## Development and evaluation dependencies

```bash
"$BEV_ROOT/.venv/bin/python" -m pip install -e '.[dev,benchmark]'
"$BEV_ROOT/.venv/bin/python" -m pytest -q
```

The `benchmark` extra pins `typesafe-sdk==0.7.1`, used by the Persian benchmark
runner. It does not download benchmark data or contact the hosted Jev API by
itself. Tests use mocked inference responses; starting the service and running
the smoke/parity checks perform actual model computation.
