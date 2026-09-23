# Third-party notices

Bev adapts the finite-choice contract, surrogate-label method and classification prompt from [Jevfire](https://github.com/kikoncuo/jevfire), revision `5df83b558bf635e006d31f8f2fc2798d0e3ff051`, copyright (c) 2026 kikoncuo, MIT. The complete MIT notice is retained in LICENSE.

Model weights are supplied by [Prism ML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, under Apache-2.0. The [Bev Hugging Face bundle](https://huggingface.co/Reza2kn/Bev) redistributes the original GGUF byte-for-byte. Bev did not train, quantize, merge, or otherwise modify it. The model derives from Qwen3.8-27B. Its complete [Apache-2.0 license](licenses/MODEL-APACHE-2.0.txt) and [original NOTICE](licenses/MODEL-NOTICE.txt) are retained; the model bundle also includes these as LICENSE and NOTICE.txt beside the weights. The GitHub repository contains code and metadata, not the weights.

The inference runtime is [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), MIT, pinned to `9a9394a895b96003ca842a6041cb28ac49a108f7` (release `prism-b10709-9a9394a`). Its [complete MIT license](licenses/PRISM-LLAMA-MIT.txt) is included with the patch. The installer downloads the separately distributed runtime and retains its included notices. The Bev patch changes the scoring API, not model weights or CUDA kernels.

The [Decision Index reproduction kit](https://github.com/apolinario/decision-index), revision `52a698928a9ae5bdf16b75687c903871db29c6e5`, is MIT. It was used in a separate source-derived diagnostic and is not vendored here. Benchmark datasets retain their separate upstream terms. Evaluation inputs are not training data and are not included in public result artifacts.

The [Jev Persian Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark), revision `ac218d96630da9d9cc08fd897868c4d3c7048b0d`, supplies the requested Persian evaluation. Its external checkout is excluded from Bev's source tree; its frozen data, runner and scorer are used unchanged. No explicit repository license was found at that revision. This project does not assert a redistribution license for its source or dataset.

Bev is an independent project, not affiliated with Jev / TypeSafe AI, Jevfire, Prism ML, Qwen, or the Decision Index maintainers.

The software in this repository is MIT-licensed; the redistributed model remains Apache-2.0. Development and documentation were assisted by OpenAI Codex. The release credits upstream work separately from Bev's runtime integration, typed API, testing, and evaluation.
