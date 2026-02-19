"""
S7-LLM-MOE-300M Training Configuration
Authority: KUHUL_π
Fold: ⟁COMPUTE_FOLD⟁

All hyperparameters in one place — no magic numbers scattered across files.
"""
from dataclasses import dataclass, field
from typing import List


# ── Architecture ─────────────────────────────────────────────────────────────

VOCAB_SIZE    = 32_768
TRUNK_HIDDEN  = 1024
TRUNK_LAYERS  = 12
TRUNK_HEADS   = 16
TRUNK_FFN_DIM = TRUNK_HIDDEN * 4   # 4096
MAX_CONTEXT   = 2048

ROUTER_HIDDEN = 2048

NUM_EXPERTS   = 9
EXPERT_HIDDEN = 1024
EXPERT_LAYERS = 4
EXPERT_HEADS  = 16
EXPERT_FFN_DIM = EXPERT_HIDDEN * 4  # 4096

# Expert-to-micronaut mapping (identity labels, specialization emerges from training).
EXPERT_NAMES = ["PM-1", "CM-1", "TM-1", "HM-1", "MM-1", "XM-1", "SM-1", "VM-2", "VM-1"]
EXPERT_FOLDS = [
    "⟁DATA_FOLD⟁", "⟁CONTROL_FOLD⟁", "⟁TIME_FOLD⟁",
    "⟁STATE_FOLD⟁", "⟁COMPUTE_FOLD⟁", "⟁PATTERN_FOLD⟁",
    "⟁STORAGE_FOLD⟁", "⟁META_FOLD⟁", "⟁UI_FOLD⟁",
]


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Loss coefficients
    alpha_balance: float = 0.01    # Load-balancing penalty weight
    beta_entropy:  float = 0.001   # Router entropy regularizer weight

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay:  float = 0.1
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0

    # Batch / context
    batch_size:    int   = 512
    seq_len:       int   = 2048
    grad_accum:    int   = 4       # effective batch = 512 * 4 = 2048 seqs

    # Schedule
    warmup_steps:  int   = 2000
    total_steps:   int   = 500_000
    lr_decay:      str   = "cosine"

    # Phase 1: dense pretrain (uniform routing, all experts active)
    phase1_steps:  int   = 200_000
    phase1_routing: str  = "uniform"

    # Phase 2: sparse routing (top-1 argmax, load-balance loss active)
    phase2_steps:  int   = 300_000
    phase2_routing: str  = "top1"

    # Checkpointing
    save_every:    int   = 5_000
    eval_every:    int   = 1_000
    checkpoint_dir: str  = "checkpoints/"

    # Precision
    dtype:         str   = "bfloat16"   # BF16 forward, FP32 gradient accum
    compile:       bool  = True          # torch.compile

    # Data curriculum
    domain_ratios: dict = field(default_factory=lambda: {
        "code":      0.35,
        "math":      0.25,
        "reasoning": 0.20,
        "general":   0.20,
    })

    # Phase 2 domain pressure (domain-specific batches routed to target experts)
    curriculum_phase: bool = True


# ── Quantisation ──────────────────────────────────────────────────────────────

@dataclass
class QuantConfig:
    method:          str   = "per_row_absmax"
    calibration_samples: int = 4096
    calibration_seq_len: int = 512
    avx2_pad:        bool  = True   # Pad to 32-byte boundary
    output_dir:      str   = "model/weights/"
    artifact_path:   str   = "model/moe-300m.s7l"
    vocab_path:      str   = "model/vocab.json"
