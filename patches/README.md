# Pinned Prism runtime extension

`prism-selected-logprobs.patch` applies to PrismML-Eng/llama.cpp commit
`9a9394a895b96003ca842a6041cb28ac49a108f7`, release `prism-b10709-9a9394a`.
Patch SHA-256: `7dadeaf5fccc2cb19c7c78e76173368168f9c904d7c88a6afe844722bba91cc2`.

The patch adds `selected_token_ids` to native `/completion`. It accepts 1–4096
unique integer token IDs within the model vocabulary and returns their raw
full-vocabulary logprobs in the given order under the existing
`completion_probabilities[...].top_logprobs` response. It auto-enables logprob
collection, rejects `post_sampling_probs:true`, and advertises
`bev_selected_token_logprobs:1` on `/props`. With no new field, existing behavior
remains unchanged.

It uses the runtime's float32 softmax, normalizing over all vocabulary entries
without sorting the vocabulary. This removes the large full-vocabulary response
and allows labels outside a small top-N result. It does not constrain generation
or normalize probabilities over the selected IDs. The original serializer's
finite sentinel for log(0) still applies. Changing summation order can create
small floating-point differences from the sorted full-vocabulary reference.

## Build without recompiling CUDA

Use the main installer on a Linux x86_64 NVIDIA host. See the
[installation guide](../docs/INSTALL.md) for dependencies and startup instructions.
Run these commands in Bash from the Bev source checkout:

```bash
source scripts/runtime-env.sh
bash "$BEV_CODE/scripts/install.sh"
```

`BEV_CODE` identifies the source checkout; `BEV_ROOT` contains downloaded weights,
runtime libraries, the patched source and build, the Python environment, and local
results. The default `BEV_ROOT` is `${XDG_DATA_HOME:-$HOME/.local/share}/bev`.
To use another installation directory, export `BEV_ROOT` before sourcing the
environment or running the installer, and keep the same value when starting Bev.

The installer checks the pinned model and runtime hashes, applies the patch to the
pinned source revision, builds `llama-server` with `GGML_CUDA=OFF`,
`GGML_BACKEND_DL=ON`, and `GGML_NATIVE=OFF`, and compiles
`bev-cuda-loader.cpp` as a shared library linked to the prebuilt CUDA backend.
CUDA kernels are supplied by the pinned CUDA 12.8 release, so this host build does
not require recompiling them. `BEV_BUILD_JOBS` controls build parallelism and
defaults to 6.

The release's prebuilt CUDA backend does not export `ggml_backend_init` because
it was built without `GGML_BACKEND_DL`. The shim exports that one generic entry
point, forwarding to the original `ggml_backend_cuda_reg`. It does not modify the
backend, create CUDA kernels, or replace backend operations. Pinning the host and
CUDA library to the same commit is mandatory. Mixing future versions is untested.

For a manual device check after installation, use the same environment as
`scripts/start-llama.sh`:

```bash
export LD_LIBRARY_PATH="$BEV_ROOT/prism-build/bin:$BEV_ROOT/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="$BEV_ROOT/prism-build/bin/libbev-cuda-loader.so"
"$BEV_ROOT/prism-build/bin/llama-server" --list-devices
```

`GGML_BACKEND_PATH` is an explicit shared-library **file**, not a directory. Put
the new host build first in `LD_LIBRARY_PATH` so its `libllama`, `libggml`, and
`libggml-base` are used; the second directory supplies the prebuilt CUDA backend.
Verify CUDA device visibility and startup layer offload before claiming GPU use.

## Semantic probe

Run after the patched server starts, using the environment sourced above:

```bash
"$BEV_ROOT/.venv/bin/python" "$BEV_CODE/scripts/probe_runtime.py" \
  --url "$BEV_LLAMA_URL" \
  --output "$BEV_ROOT/artifacts/runtime-probe.json"
```

This makes three one-token inference calls: full-vocabulary reference, a selected
255-ID request including an ID outside top-512 in a different order, and a selected
ID request with strong sampler bias. It checks raw-score parity within 0.0003
absolute logprob, full-vocabulary probability mass, selected output order, and
seven rejected malformed requests. It saves timing and bounded provenance, not
the full-vocabulary JSON. This validates the extension against its own runtime;
it does not establish FP16, Jevfire, or vLLM equivalence.

Verified on an RTX 5080 Laptop GPU: all 255 requested IDs matched the full 248,320-token
reference with maximum absolute logprob difference `9.5367431640625e-07`.
Applying a strong sampler bias preserved raw scores to the same tolerance.
All seven malformed requests returned HTTP 400. The three reference/selected/
biased requests each processed a cold 41-token prompt with zero cached tokens.

The native full-vocabulary probabilities summed to `1.0002496007300135` when
re-summed in Python double precision. The initial mass tolerance `1e-4` was too
tight for the upstream float32 reduction; the probe now allows `1e-3` and records
the observed deviation rather than hiding or renormalizing it. This change does
not loosen the independent `0.0003` logprob parity bound. See the
[published runtime probe](../evaluations/runtime-probe.json) for the recorded result.
