"""
S7-LLM-MINI training configuration.

Architecture targets ~800K parameters:
  vocab=128 (ASCII byte-level, fits in i8)  d_model=128  n_heads=4  n_layers=4  ffn_dim=512
  → embedding: 128×128 = 16K (weight-tied with lm_head)
  → 4 × (attn 4×128² + ffn 2×128×512) = 786K
  → total ≈ 802K
"""
from dataclasses import dataclass, field


@dataclass
class MiniConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size:  int = 128    # ASCII byte-level; IDs 0-127 fit in Rust's i8 tokenizer
    d_model:     int = 128
    n_heads:     int = 4      # head_dim = 32
    n_layers:    int = 4
    ffn_dim:     int = 512
    max_seq_len: int = 128

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    batch_size:     int   = 256
    max_steps:      int   = 50_000
    warmup_steps:   int   = 1_000
    grad_clip:      float = 1.0
    log_every:      int   = 100
    ckpt_every:     int   = 5_000

    # ── Quantisation ──────────────────────────────────────────────────────────
    # Number of calibration batches for per-row absmax scale estimation.
    calib_batches: int = 200
    # KL divergence threshold between FP32 and INT8 logits (per token, nats).
    kl_threshold:  float = 0.05

    # ── I/O ───────────────────────────────────────────────────────────────────
    out_dir:       str = "out/"
    ckpt_path:     str = "out/mini.pt"
    weights_dir:   str = "out/weights/"
    vocab_out:     str = "out/vocab.json"
    s7l_out:       str = "../model/mini.s7l"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_budget(self) -> dict:
        emb   = self.vocab_size * self.d_model
        attn  = 4 * (self.d_model ** 2)   # q+k+v+o per layer
        ffn   = 2 * self.d_model * self.ffn_dim  # fc1+fc2 per layer
        layer = attn + ffn
        total = emb + self.n_layers * layer
        return {
            "embedding":   emb,
            "attn/layer":  attn,
            "ffn/layer":   ffn,
            "n_layers":    self.n_layers,
            "total":       total,
        }
