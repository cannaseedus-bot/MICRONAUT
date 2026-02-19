"""
S7-LLM-Mini PyTorch model for training.

Architecture: 4-layer causal transformer, hidden=128, ReLU FFN.
Matches the Rust inference engine (attention.rs + ffn.rs + transformer.rs).

Token IDs: byte-level ASCII 0-127 (fit in Rust's i8 tokenizer).
Weight tying: embedding.weight == lm_head.weight.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniConfig


# ── Rotary Positional Embedding ───────────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple:
    """Precompute cos/sin tables for RoPE."""
    theta = 10000.0 ** (-2 * torch.arange(0, head_dim // 2, device=device) / head_dim)
    t     = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, theta)                  # [T, head_dim/2]
    cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, head_dim]
    sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor [B, T, H, head_dim]."""
    return x * cos.unsqueeze(0).unsqueeze(2) + rotate_half(x) * sin.unsqueeze(0).unsqueeze(2)


# ── Causal Self-Attention ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model

        self.q_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x:   torch.Tensor,         # [B, T, D]
        cos: torch.Tensor,         # [T, head_dim]
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh)
        k = self.k_proj(x).view(B, T, H, Dh)
        v = self.v_proj(x).view(B, T, H, Dh)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # [B, H, T, Dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask.
        scale  = math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]
        mask   = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        out = torch.matmul(weights, v)              # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ── Feed-Forward Network ───────────────────────────────────────────────────────

class FFN(nn.Module):
    """Two-layer FFN with ReLU — matches Rust FFN (ffn.rs uses relu, not GELU)."""
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN → Attn → residual → LN → FFN → residual."""
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


# ── Top-level Model ───────────────────────────────────────────────────────────

class S7MiniModel(nn.Module):
    """
    S7-LLM-Mini: causal language model, ~800K parameters.

    Token IDs are in [0, vocab_size) and represent raw ASCII bytes.
    Weight tying: lm_head.weight = embedding.weight (no extra parameters).
    """
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_out    = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying — lm_head shares embedding matrix.
        self.lm_head.weight = self.embedding.weight

        # RoPE cache — extended lazily in forward if needed.
        self._rope_len = 0
        self._cos      = None
        self._sin      = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.q_proj.weight, std=0.02)
            nn.init.normal_(block.attn.k_proj.weight, std=0.02)
            nn.init.normal_(block.attn.v_proj.weight, std=0.02)
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            nn.init.normal_(block.ffn.fc1.weight, std=0.02)
            nn.init.normal_(block.ffn.fc2.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def _get_rope(self, T: int, device: torch.device):
        if T > self._rope_len:
            self._cos, self._sin = build_rope_cache(
                max(T, self.cfg.max_seq_len), self.cfg.head_dim, device
            )
            self._rope_len = self._cos.shape[0]
        return self._cos[:T], self._sin[:T]

    def forward(
        self,
        tokens: torch.Tensor,         # [B, T]  long, values in [0, vocab_size)
        targets: torch.Tensor = None, # [B, T]  long — if given, compute LM loss
    ) -> dict:
        B, T = tokens.shape
        cos, sin = self._get_rope(T, tokens.device)

        x = self.embedding(tokens)                 # [B, T, D]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_out(x)
        logits = self.lm_head(x)                   # [B, T, vocab_size]

        out = {"logits": logits}
        if targets is not None:
            # Shift: predict token t+1 from context t.
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.cfg.vocab_size),
                targets[:, 1:].reshape(-1),
                ignore_index=-1,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new: int = 64) -> list[int]:
        """Greedy argmax decode — mirrors the Rust inference engine."""
        self.eval()
        ids = list(prompt_ids)
        device = next(self.parameters()).device
        for _ in range(max_new):
            ctx = ids[-self.cfg.max_seq_len:]
            inp = torch.tensor([ctx], dtype=torch.long, device=device)
            out = self.forward(inp)
            next_id = int(out["logits"][0, -1].argmax())
            if next_id == 0:   # 0 = <unk> / EOS
                break
            ids.append(next_id)
        return ids
