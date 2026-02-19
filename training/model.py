"""
S7-LLM-MOE-300M: PyTorch model definition.

Architecture:
    SharedTrunk (80M)    — 12-layer transformer, hidden=1024
    LearnedRouter  (2M)  — MLP: 1024→2048→9, trained end-to-end
    9 Experts (216M)     — 4-layer transformers, hidden=1024, fold-bound
    LM Head              — weight-tied with embedding

Training routing: top-1 argmax (or uniform in Phase 1).
Inference: always argmax (deterministic).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import (
    VOCAB_SIZE, TRUNK_HIDDEN, TRUNK_LAYERS, TRUNK_HEADS, TRUNK_FFN_DIM,
    ROUTER_HIDDEN, NUM_EXPERTS, EXPERT_HIDDEN, EXPERT_LAYERS, EXPERT_HEADS,
    EXPERT_FFN_DIM, MAX_CONTEXT, EXPERT_NAMES, EXPERT_FOLDS,
)


# ── Rotary Positional Encoding ────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_pos: int = MAX_CONTEXT):
        super().__init__()
        inv_freq = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.head_dim = head_dim

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)            # [seq, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [seq, head_dim]
        return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ── Multi-Head Causal Self-Attention ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.rope     = RotaryEmbedding(self.head_dim)

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention with causal mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


# ── Feed-Forward Network (GELU) ───────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, hidden: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# ── Transformer Layer (Pre-LN) ────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = CausalSelfAttention(hidden, n_heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn   = FFN(hidden, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Shared Trunk ──────────────────────────────────────────────────────────────

class SharedTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, TRUNK_HIDDEN)
        self.layers = nn.ModuleList([
            TransformerLayer(TRUNK_HIDDEN, TRUNK_HEADS, TRUNK_FFN_DIM)
            for _ in range(TRUNK_LAYERS)
        ])
        self.norm = nn.LayerNorm(TRUNK_HIDDEN)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, T] → hidden: [B, T, TRUNK_HIDDEN]"""
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ── Learned Router (MLP) ──────────────────────────────────────────────────────

class LearnedRouter(nn.Module):
    """
    Router MLP: TRUNK_HIDDEN → ROUTER_HIDDEN → NUM_EXPERTS.

    During training: returns softmax probabilities (used for L_balance, L_entropy).
    During inference: argmax of logits (deterministic, no softmax needed).
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(TRUNK_HIDDEN, ROUTER_HIDDEN, bias=True)
        self.fc2 = nn.Linear(ROUTER_HIDDEN, NUM_EXPERTS, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., TRUNK_HIDDEN] → logits: [..., NUM_EXPERTS]"""
        return self.fc2(F.gelu(self.fc1(h), approximate="tanh"))

    @torch.no_grad()
    def route_deterministic(self, h: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns expert_id = argmax(logits). No gradients."""
        return self.forward(h).argmax(dim=-1)


# ── Expert Subnetwork ─────────────────────────────────────────────────────────

class Expert(nn.Module):
    """
    One expert: 4-layer transformer, hidden=1024.
    Fold-bound identity stored as metadata (not enforced at training time).
    Specialization emerges from routing + load-balance objective.
    """
    def __init__(self, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.micronaut = EXPERT_NAMES[expert_id]
        self.fold      = EXPERT_FOLDS[expert_id]
        self.layers = nn.ModuleList([
            TransformerLayer(EXPERT_HIDDEN, EXPERT_HEADS, EXPERT_FFN_DIM)
            for _ in range(EXPERT_LAYERS)
        ])
        self.norm = nn.LayerNorm(EXPERT_HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B_e, T, EXPERT_HIDDEN] → [B_e, T, EXPERT_HIDDEN]"""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ── Full MoE Model ────────────────────────────────────────────────────────────

class S7LlmMoe300M(nn.Module):
    """
    S7-LLM-MOE-300M: SharedTrunk + LearnedRouter + 9 Experts + LM Head.

    Forward pass produces:
        logits        [B, T, VOCAB_SIZE]
        router_probs  [B, T, NUM_EXPERTS]   (for loss computation)
        expert_ids    [B, T]                (int64, which expert was chosen)
        routing_mode  str                   ("uniform" or "top1")
    """
    def __init__(self, routing_mode: str = "top1"):
        super().__init__()
        self.routing_mode = routing_mode

        self.trunk   = SharedTrunk()
        self.router  = LearnedRouter()
        self.experts = nn.ModuleList([Expert(i) for i in range(NUM_EXPERTS)])
        self.lm_head = nn.Linear(EXPERT_HIDDEN, VOCAB_SIZE, bias=False)

        # Weight tying: embedding and lm_head share the same matrix.
        self.lm_head.weight = self.trunk.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * TRUNK_LAYERS))
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        tokens: torch.Tensor,   # [B, T]
        labels: Optional[torch.Tensor] = None,  # [B, T] for loss
    ):
        B, T = tokens.shape

        # 1. Shared trunk.
        h = self.trunk(tokens)   # [B, T, 1024]

        # 2. Router logits and probabilities.
        router_logits = self.router(h)                         # [B, T, 9]
        router_probs  = F.softmax(router_logits, dim=-1)       # [B, T, 9]

        # 3. Expert dispatch (top-1 routing or uniform).
        if self.routing_mode == "uniform":
            # Phase 1: all experts contribute equally (dense training).
            expert_out = torch.stack(
                [self.experts[i](h) for i in range(NUM_EXPERTS)], dim=-1
            ).mean(dim=-1)
            expert_ids = router_probs.argmax(dim=-1)           # for logging only
        else:
            # Phase 2: top-1 argmax routing.
            expert_ids = router_probs.argmax(dim=-1)           # [B, T]
            expert_out = self._dispatch_top1(h, expert_ids)    # [B, T, 1024]

        # 4. LM head.
        logits = self.lm_head(expert_out)   # [B, T, VOCAB_SIZE]

        return {
            "logits":        logits,
            "router_probs":  router_probs,
            "router_logits": router_logits,
            "expert_ids":    expert_ids,
        }

    def _dispatch_top1(self, h: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Route each token to its selected expert.
        Tokens are gathered per expert, processed in batch, then scattered back.

        h:          [B, T, hidden]
        expert_ids: [B, T]         int64
        Returns:    [B, T, hidden]
        """
        B, T, D = h.shape
        h_flat  = h.view(B * T, D)         # [B*T, hidden]
        ids_flat = expert_ids.view(B * T)  # [B*T]
        out_flat = torch.zeros_like(h_flat)

        for eid in range(NUM_EXPERTS):
            mask = (ids_flat == eid)        # [B*T] bool
            if mask.any():
                tokens_e = h_flat[mask]     # [n_e, hidden]
                # Unsqueeze/squeeze to give expert a [1, n_e, hidden] batch.
                out_e = self.experts[eid](tokens_e.unsqueeze(0)).squeeze(0)
                out_flat[mask] = out_e

        return out_flat.view(B, T, D)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expert_param_count(self) -> int:
        return sum(p.numel() for e in self.experts for p in e.parameters())
