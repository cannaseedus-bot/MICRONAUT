/// WebGPU runtime for S7-LLM-MOE-140M.
///
/// Feature flag: `--features webgpu`
/// Dependency:   `wgpu` crate (not included in default Cargo.toml to keep
///               the CPU path dependency-free).
///
/// The WebGPU path uses the WGSL shaders in `shaders/` to execute:
///   1. Embedding lookup         (embedding.wgsl)
///   2. INT8 matmul              (matmul_int8.wgsl)
///   3. Attention                (attention.wgsl)
///   4. FFN + GELU               (ffn.wgsl)
///   5. Router (identity copy)   (router.wgsl)
///
/// Memory budget (INT8, context 2048, 4 experts):
///   Model weights : ~140MB  (fits in WebGPU device memory)
///   KV cache      : ~36MB   (trunk + 1 active expert)
///   Activations   : ~4MB    (double-buffered per layer)
///   Total         : ~180MB
pub mod device;
pub mod pipeline;
