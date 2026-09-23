// ABI bridge for Prism llama.cpp 9a9394a895b96003ca842a6041cb28ac49a108f7.
// Its CUDA 12.8 release backend exports ggml_backend_cuda_reg but was not built
// with GGML_BACKEND_DL, so it does not expose the generic dynamic entry point.
// Build this against that release only; no CUDA compilation is involved.
// The backend registry is opaque here and its address is passed unchanged.
extern "C" {
struct ggml_backend_reg;
ggml_backend_reg * ggml_backend_cuda_reg(void);

__attribute__((visibility("default")))
ggml_backend_reg * ggml_backend_init(void) {
    return ggml_backend_cuda_reg();
}
}
