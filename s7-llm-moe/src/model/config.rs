/// Static architecture constants for S7-LLM-MOE-300M.
///
/// 9-expert learned MoE — one expert per micronaut/fold domain.
/// Router is a learned MLP (INT8 at inference, FP32 at training).
/// Inference is deterministic: argmax over router logits (no sampling).

// ─── Vocabulary ──────────────────────────────────────────────────────────────
pub const VOCAB_SIZE:    usize = 32_768;  // 32k, divisible by 32

// ─── Shared Trunk ─────────────────────────────────────────────────────────────
pub const TRUNK_HIDDEN:   usize = 1024;
pub const TRUNK_LAYERS:   usize = 12;
pub const TRUNK_HEADS:    usize = 16;
pub const TRUNK_HEAD_DIM: usize = TRUNK_HIDDEN / TRUNK_HEADS; // 64
pub const TRUNK_FFN_MUL:  usize = 4;
pub const TRUNK_FFN_DIM:  usize = TRUNK_HIDDEN * TRUNK_FFN_MUL; // 4096
pub const MAX_CONTEXT:    usize = 2048;

// ─── Learned Router MLP ──────────────────────────────────────────────────────
/// Architecture:
///   Linear(TRUNK_HIDDEN → ROUTER_HIDDEN) → GELU → Linear(ROUTER_HIDDEN → NUM_EXPERTS)
///   At inference: argmax over output logits (deterministic).
pub const ROUTER_HIDDEN:  usize = 2048;   // MLP bottleneck width

// ─── Experts ─────────────────────────────────────────────────────────────────
pub const NUM_EXPERTS:    usize = 9;
pub const EXPERT_HIDDEN:  usize = 1024;
pub const EXPERT_LAYERS:  usize = 4;
pub const EXPERT_HEADS:   usize = 16;
pub const EXPERT_HEAD_DIM: usize = EXPERT_HIDDEN / EXPERT_HEADS; // 64
pub const EXPERT_FFN_MUL: usize = 4;
pub const EXPERT_FFN_DIM: usize = EXPERT_HIDDEN * EXPERT_FFN_MUL; // 4096

// Trunk and expert hidden dims are equal — no projection needed.
pub const PROJ_IN_DIM:    usize = TRUNK_HIDDEN;
pub const PROJ_OUT_DIM:   usize = EXPERT_HIDDEN;

// ─── Expert-to-Micronaut Mapping (fold-scoped) ───────────────────────────────
/// Expert indices match micronaut registration order.
/// Specialization emerges from training — these are identity labels only.
/// The router selects the expert; it is NOT pre-assigned by domain.
pub const EXPERT_PM1: usize = 0;  // Perception   ⟁DATA_FOLD⟁
pub const EXPERT_CM1: usize = 1;  // Control      ⟁CONTROL_FOLD⟁
pub const EXPERT_TM1: usize = 2;  // Temporal     ⟁TIME_FOLD⟁
pub const EXPERT_HM1: usize = 3;  // Host         ⟁STATE_FOLD⟁
pub const EXPERT_MM1: usize = 4;  // Compute      ⟁COMPUTE_FOLD⟁
pub const EXPERT_XM1: usize = 5;  // Pattern      ⟁PATTERN_FOLD⟁
pub const EXPERT_SM1: usize = 6;  // Storage      ⟁STORAGE_FOLD⟁
pub const EXPERT_VM2: usize = 7;  // Verification ⟁META_FOLD⟁
pub const EXPERT_VM1: usize = 8;  // Projection   ⟁UI_FOLD⟁

pub const EXPERT_NAMES: [&str; NUM_EXPERTS] = [
    "PM-1", "CM-1", "TM-1", "HM-1", "MM-1", "XM-1", "SM-1", "VM-2", "VM-1",
];

pub const EXPERT_FOLDS: [&str; NUM_EXPERTS] = [
    "⟁DATA_FOLD⟁",
    "⟁CONTROL_FOLD⟁",
    "⟁TIME_FOLD⟁",
    "⟁STATE_FOLD⟁",
    "⟁COMPUTE_FOLD⟁",
    "⟁PATTERN_FOLD⟁",
    "⟁STORAGE_FOLD⟁",
    "⟁META_FOLD⟁",
    "⟁UI_FOLD⟁",
];

// ─── Training Hyperparameters (reference values, not enforced at runtime) ─────
pub const ALPHA_BALANCE: f32 = 0.01;    // Load-balancing loss scale
pub const BETA_ENTROPY:  f32 = 0.001;   // Router entropy regularizer scale

// ─── Approximate Parameter Budget ────────────────────────────────────────────
/// Trunk:       32768*1024 + 12*(4*1024²+2*1024*4096)      ≈ 80M
/// Router:      1024*2048  + 2048*9                         ≈  2M
/// Experts:     9 * 4*(4*1024²+2*1024*4096)                ≈ 216M
/// Total active per token:  trunk(80M) + one expert(24M)   ≈ 104M
pub const APPROX_TOTAL_PARAMS_M:  usize = 300;
pub const APPROX_ACTIVE_PARAMS_M: usize = 104;

// ─── Inference ────────────────────────────────────────────────────────────────
pub const DEFAULT_MAX_TOKENS: usize = 512;
pub const ACCUM_SHIFT:        i32   = 7;
