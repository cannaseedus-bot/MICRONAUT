"""
S7-LLM-MOE-300M Training Loop.

Two-phase training:
    Phase 1: Dense pretrain (routing mode=uniform, no auxiliary losses).
             Trunk + all experts trained uniformly. Router present but loss-free.
    Phase 2: Sparse MoE (routing mode=top1, auxiliary losses active).
             Spontaneous expert specialization via routing + load-balance pressure.

Usage:
    # Full training run
    python train.py --config default --output checkpoints/moe-300m/

    # Resume from checkpoint
    python train.py --resume checkpoints/moe-300m/step_100000/

    # Phase 2 only (from phase 1 checkpoint)
    python train.py --phase 2 --resume checkpoints/moe-300m/phase1_final/
"""
import os
import math
import time
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from config import TrainConfig, NUM_EXPERTS, EXPERT_NAMES
from model import S7LlmMoe300M
from losses import moe_loss, ExpertMonitor
from data import CurriculumLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Learning Rate Schedule ─────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Cosine decay with linear warmup.
        step < warmup_steps  → linear ramp 0 → lr
        step >= warmup_steps → cosine decay lr → 0
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW with weight decay applied only to ≥2D tensors (skip biases/norms).
    """
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=torch.cuda.is_available(),
    )


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, phase, cfg, output_dir):
    ckpt_dir = Path(output_dir) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":       step,
        "phase":      phase,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "config":     cfg.__dict__,
    }, ckpt_dir / "checkpoint.pt")
    log.info("Checkpoint saved: %s", ckpt_dir)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0), ckpt.get("phase", 1)


# ── Training Phase 1: Dense Pretrain ─────────────────────────────────────────

def run_phase1(
    model: S7LlmMoe300M,
    optimizer: torch.optim.Optimizer,
    loader: CurriculumLoader,
    cfg: TrainConfig,
    output_dir: str,
    start_step: int = 0,
    device: torch.device = torch.device("cuda"),
):
    """
    Phase 1: all experts active (uniform routing), no auxiliary losses.
    Router learns basic projection; experts begin to diverge slightly.
    """
    log.info("=== Phase 1: Dense pretrain for %d steps ===", cfg.phase1_steps)
    model.routing_mode = "uniform"
    model.train()

    scaler  = GradScaler(enabled=(cfg.dtype == "bfloat16"))
    monitor = ExpertMonitor(NUM_EXPERTS)

    for step in range(start_step, cfg.phase1_steps):
        t0 = time.perf_counter()

        # Gradient accumulation loop.
        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0

        for _accum in range(cfg.grad_accum):
            tokens, labels, domain = next(loader)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=(cfg.dtype == "bfloat16")):
                out    = model(tokens, labels)
                loss, diag = moe_loss(
                    out["logits"], labels,
                    out["router_probs"], out["expert_ids"],
                    NUM_EXPERTS, phase="phase1",
                )
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()
            total_loss_val += diag["lm_loss"] / cfg.grad_accum
            monitor.update(out["expert_ids"])

        # LR update.
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        dt = time.perf_counter() - t0

        if step % 100 == 0:
            util = monitor.utilization()
            imbal = monitor.max_imbalance()
            log.info(
                "Phase1 step=%d lr=%.2e loss=%.4f dt=%.2fs imbalance=%.3f",
                step, lr, total_loss_val, dt, imbal,
            )
            monitor.reset()

        if step % cfg.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, 1, cfg, output_dir)

    save_checkpoint(model, optimizer, cfg.phase1_steps, 1, cfg, output_dir)
    log.info("Phase 1 complete.")


# ── Training Phase 2: Sparse Routing ─────────────────────────────────────────

def run_phase2(
    model: S7LlmMoe300M,
    optimizer: torch.optim.Optimizer,
    loader: CurriculumLoader,
    cfg: TrainConfig,
    output_dir: str,
    start_step: int = 0,
    device: torch.device = torch.device("cuda"),
):
    """
    Phase 2: top-1 routing active, auxiliary losses on.
    Expert specialization emerges via symmetry breaking + routing pressure.
    """
    log.info("=== Phase 2: Sparse routing for %d steps ===", cfg.phase2_steps)
    model.routing_mode = "top1"
    model.train()

    scaler  = GradScaler(enabled=(cfg.dtype == "bfloat16"))
    monitor = ExpertMonitor(NUM_EXPERTS)
    best_imbalance = float("inf")

    for step in range(start_step, cfg.phase2_steps):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        total_loss_val = 0.0
        total_bal_val  = 0.0
        total_ent_val  = 0.0

        for _accum in range(cfg.grad_accum):
            tokens, labels, domain = next(loader)
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=(cfg.dtype == "bfloat16")):
                out = model(tokens, labels)
                loss, diag = moe_loss(
                    out["logits"], labels,
                    out["router_probs"], out["expert_ids"],
                    NUM_EXPERTS,
                    alpha=cfg.alpha_balance,
                    beta=cfg.beta_entropy,
                    phase="phase2",
                )
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()
            total_loss_val += diag["lm_loss"]       / cfg.grad_accum
            total_bal_val  += diag["balance_loss"]  / cfg.grad_accum
            total_ent_val  += diag["entropy_loss"]  / cfg.grad_accum
            monitor.update(out["expert_ids"])

        lr = get_lr(cfg.phase1_steps + step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        dt = time.perf_counter() - t0

        if step % 100 == 0:
            util  = monitor.utilization()
            imbal = monitor.max_imbalance()
            # Log per-expert utilization as a one-liner.
            util_str = " ".join(f"{EXPERT_NAMES[i]}={util.get(f'expert_{i}_util', 0):.2%}"
                                for i in range(NUM_EXPERTS))
            log.info(
                "Phase2 step=%d lr=%.2e lm=%.4f bal=%.4f ent=%.4f imbal=%.3f dt=%.2fs",
                step, lr, total_loss_val, total_bal_val, total_ent_val, imbal, dt,
            )
            log.info("Expert utilization: %s", util_str)
            monitor.reset()

            # Early expert-collapse detection.
            if imbal > 0.5:
                log.warning("HIGH ROUTING IMBALANCE (%.2f) — consider increasing alpha", imbal)

        if step % cfg.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, 2, cfg, output_dir)

    save_checkpoint(model, optimizer, cfg.phase2_steps, 2, cfg, output_dir)
    log.info("Phase 2 complete.")


# ── Specialization Diagnostics ────────────────────────────────────────────────

@torch.no_grad()
def log_expert_specialization(
    model: S7LlmMoe300M,
    domain_samples: dict,  # domain → (tokens, labels) tensors
    device: torch.device,
):
    """
    For each domain, pass a few batches through the model and record
    which experts are most activated.  Log the domain-expert affinity matrix.
    """
    model.eval()
    model.routing_mode = "top1"
    affinity = {domain: [0] * NUM_EXPERTS for domain in domain_samples}

    for domain, (tokens, labels) in domain_samples.items():
        tokens = tokens.to(device)
        out    = model(tokens)
        ids    = out["expert_ids"].view(-1).cpu().tolist()
        for eid in ids:
            affinity[domain][eid] += 1
        total = len(ids)
        aff_str = " ".join(
            f"{EXPERT_NAMES[i]}={affinity[domain][i]/total:.2%}"
            for i in range(NUM_EXPERTS)
        )
        log.info("Specialization %-10s → %s", domain, aff_str)

    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="checkpoints/moe-300m/")
    parser.add_argument("--resume",  default=None)
    parser.add_argument("--phase",   type=int, default=0,   # 0 = run both
                        help="1=phase1 only, 2=phase2 only, 0=both")
    parser.add_argument("--data-dir", default="data/tokenized/")
    args = parser.parse_args()

    cfg    = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Data ──
    domain_paths = {
        "code":      f"{args.data_dir}/code.bin",
        "math":      f"{args.data_dir}/math.bin",
        "reasoning": f"{args.data_dir}/reasoning.bin",
        "general":   f"{args.data_dir}/general.bin",
    }
    loader = CurriculumLoader(
        domain_paths    = domain_paths,
        domain_ratios   = cfg.domain_ratios,
        seq_len         = cfg.seq_len,
        batch_size      = cfg.batch_size,
        curriculum_step = 100 if cfg.curriculum_phase else None,
    )

    # ── Model ──
    model = S7LlmMoe300M(routing_mode="uniform").to(device)
    if cfg.compile:
        try:
            model = torch.compile(model)
            log.info("torch.compile: OK")
        except Exception as e:
            log.warning("torch.compile failed: %s", e)

    log.info("Model parameters: %dM", model.param_count() // 1_000_000)

    # ── Optimizer ──
    optimizer  = build_optimizer(model, cfg)
    start_step = 0
    start_phase = 1

    if args.resume:
        ckpt_path = Path(args.resume) / "checkpoint.pt"
        start_step, start_phase = load_checkpoint(str(ckpt_path), model, optimizer)
        log.info("Resumed from %s at step=%d phase=%d", args.resume, start_step, start_phase)

    # ── Phase 1 ──
    if args.phase in (0, 1) and start_phase == 1:
        run_phase1(model, optimizer, loader, cfg, args.output, start_step, device)
        start_step = 0

    # ── Phase 2 ──
    if args.phase in (0, 2):
        run_phase2(model, optimizer, loader, cfg, args.output, start_step, device)

    log.info("Training complete. Saved to %s", args.output)


if __name__ == "__main__":
    main()
