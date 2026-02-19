"""
S7-LLM-MOE-300M Training Losses.

Total loss:
    L_total = L_lm + α * L_balance + β * L_entropy

Components:
    L_lm       — cross-entropy language modeling loss
    L_balance  — Switch Transformer load-balancing penalty
    L_entropy  — router entropy regularizer

Reference:
    Switch Transformers (Fedus et al., 2021)
    GShard (Lepikhin et al., 2020)
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


# ── Language Modeling Loss ────────────────────────────────────────────────────

def lm_loss(
    logits: torch.Tensor,   # [B, T, vocab_size]
    labels: torch.Tensor,   # [B, T]  int64, -100 = ignore
) -> torch.Tensor:
    """
    Standard causal language modeling cross-entropy.
    Shift logits and labels by one position (predict next token).
    """
    # Shift: logits[t] predicts labels[t+1].
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, T-1]

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ── Load-Balancing Loss (Switch Transformer) ──────────────────────────────────

def balance_loss(
    router_probs: torch.Tensor,  # [B, T, E]  softmax probs
    expert_ids:   torch.Tensor,  # [B, T]     int64 argmax
    num_experts:  int,
) -> torch.Tensor:
    """
    Switch Transformer auxiliary loss to prevent expert collapse.

        L_balance = E * Σ_i (f_i * g_i)

    f_i = fraction of tokens routed to expert i    (actual)
    g_i = mean router probability for expert i     (expected)

    Minimized when f_i ≈ g_i ≈ 1/E for all experts.
    """
    B, T, E = router_probs.shape
    N = B * T  # total token count

    ids_flat   = expert_ids.view(N)          # [N]
    probs_flat = router_probs.view(N, E)     # [N, E]

    # f_i: actual fraction of tokens routed to expert i.
    f = torch.zeros(E, device=router_probs.device)
    for eid in range(E):
        f[eid] = (ids_flat == eid).float().sum() / N

    # g_i: mean softmax probability for expert i.
    g = probs_flat.mean(dim=0)               # [E]

    return num_experts * torch.sum(f * g)


# ── Router Entropy Regularizer ────────────────────────────────────────────────

def entropy_loss(
    router_probs: torch.Tensor,   # [B, T, E]  softmax probs
) -> torch.Tensor:
    """
    Negative entropy of routing distribution.
    Encourages router to remain sharp (low entropy = confident routing).
    Small positive β means we add a mild sharpness pressure.

        L_entropy = - mean_over_tokens( Σ_i p_i * log(p_i) )

    Note: we return the negative entropy so that minimizing L_entropy
    increases routing sharpness.  Scale by small β (0.001).
    """
    # Clamp to avoid log(0).
    p = router_probs.clamp(min=1e-9)
    return -(p * p.log()).sum(dim=-1).mean()


# ── Combined MoE Loss ─────────────────────────────────────────────────────────

def moe_loss(
    logits:        torch.Tensor,  # [B, T, vocab_size]
    labels:        torch.Tensor,  # [B, T]
    router_probs:  torch.Tensor,  # [B, T, E]
    expert_ids:    torch.Tensor,  # [B, T]
    num_experts:   int,
    alpha:         float = 0.01,
    beta:          float = 0.001,
    phase:         str   = "phase2",   # "phase1" disables auxiliary losses
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full training loss with diagnostics dict.

    Returns:
        loss      — scalar tensor (backprop through this)
        diag      — {lm_loss, balance_loss, entropy_loss, total_loss} as floats
    """
    L_lm = lm_loss(logits, labels)

    if phase == "phase1":
        # Dense pre-training: no auxiliary losses (router not yet active).
        diag = {
            "lm_loss":       L_lm.item(),
            "balance_loss":  0.0,
            "entropy_loss":  0.0,
            "total_loss":    L_lm.item(),
        }
        return L_lm, diag

    L_bal  = balance_loss(router_probs, expert_ids, num_experts)
    L_ent  = entropy_loss(router_probs)
    L_total = L_lm + alpha * L_bal + beta * L_ent

    diag = {
        "lm_loss":       L_lm.item(),
        "balance_loss":  L_bal.item(),
        "entropy_loss":  L_ent.item(),
        "total_loss":    L_total.item(),
    }
    return L_total, diag


# ── Expert Utilization Monitor ────────────────────────────────────────────────

class ExpertMonitor:
    """
    Track expert selection statistics across training batches.
    Used to detect expert collapse early (one expert receiving all tokens).
    """
    def __init__(self, num_experts: int):
        self.num_experts  = num_experts
        self.token_counts = [0] * num_experts
        self.total_tokens = 0

    def update(self, expert_ids: torch.Tensor):
        ids = expert_ids.view(-1).cpu().tolist()
        for eid in ids:
            self.token_counts[eid] += 1
        self.total_tokens += len(ids)

    def utilization(self) -> Dict[str, float]:
        if self.total_tokens == 0:
            return {}
        return {
            f"expert_{i}_util": self.token_counts[i] / self.total_tokens
            for i in range(self.num_experts)
        }

    def max_imbalance(self) -> float:
        """Max deviation from uniform routing (0 = perfectly balanced)."""
        if self.total_tokens == 0:
            return 0.0
        uniform = 1.0 / self.num_experts
        utils = [c / self.total_tokens for c in self.token_counts]
        return max(abs(u - uniform) for u in utils)

    def reset(self):
        self.token_counts = [0] * self.num_experts
        self.total_tokens = 0
