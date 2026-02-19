/// Static architecture constants for S7-LLM-MOE-140M.
/// These are compile-time law — not runtime configuration.

// ─── Vocabulary ──────────────────────────────────────────────────────────────
/// BPE vocabulary size (24576 = 24k, divisible by 32 for AVX2 alignment).
pub const VOCAB_SIZE: usize = 24_576;

// ─── Shared Trunk ─────────────────────────────────────────────────────────────
/// Trunk hidden dimension.
pub const TRUNK_HIDDEN: usize = 768;
/// Number of trunk transformer layers.
pub const TRUNK_LAYERS: usize = 6;
/// Number of attention heads in trunk.
pub const TRUNK_HEADS: usize = 12;
/// Head dimension = TRUNK_HIDDEN / TRUNK_HEADS.
pub const TRUNK_HEAD_DIM: usize = TRUNK_HIDDEN / TRUNK_HEADS; // 64
/// FFN expansion factor for trunk.
pub const TRUNK_FFN_MUL: usize = 4;
/// Trunk FFN intermediate dimension.
pub const TRUNK_FFN_DIM: usize = TRUNK_HIDDEN * TRUNK_FFN_MUL; // 3072
/// Maximum context length.
pub const MAX_CONTEXT: usize = 2048;

// ─── Expert Layers ───────────────────────────────────────────────────────────
/// Expert hidden dimension (larger than trunk for capacity).
pub const EXPERT_HIDDEN: usize = 512;
/// Number of transformer layers per expert.
pub const EXPERT_LAYERS: usize = 8;
/// Number of attention heads per expert.
pub const EXPERT_HEADS: usize = 8;
/// Head dimension for experts.
pub const EXPERT_HEAD_DIM: usize = EXPERT_HIDDEN / EXPERT_HEADS; // 64
/// FFN expansion for experts.
pub const EXPERT_FFN_MUL: usize = 4;
/// Expert FFN intermediate dimension.
pub const EXPERT_FFN_DIM: usize = EXPERT_HIDDEN * EXPERT_FFN_MUL; // 2048

// ─── Projection (trunk → expert hidden) ─────────────────────────────────────
/// Linear projection from trunk output to expert input.
/// Shape: [TRUNK_HIDDEN, EXPERT_HIDDEN]
pub const PROJ_IN_DIM:  usize = TRUNK_HIDDEN;
pub const PROJ_OUT_DIM: usize = EXPERT_HIDDEN;

// ─── MoE Routing ─────────────────────────────────────────────────────────────
/// Number of experts.
pub const NUM_EXPERTS: usize = 4;
/// Expert indices.
pub const EXPERT_CODE:   usize = 0;
pub const EXPERT_MATH:   usize = 1;
pub const EXPERT_REASON: usize = 2;
pub const EXPERT_GENERAL: usize = 3;

// ─── Inference ────────────────────────────────────────────────────────────────
/// Default maximum generation length.
pub const DEFAULT_MAX_TOKENS: usize = 512;

// ─── Quantisation ────────────────────────────────────────────────────────────
/// INT8 accumulator right-shift to normalise into i8 range.
pub const ACCUM_SHIFT: i32 = 7;
