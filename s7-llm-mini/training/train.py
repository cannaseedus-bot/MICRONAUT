"""
S7-LLM-Mini training loop.

Data: raw text files; encoded as ASCII bytes (0-127).
      Any byte ≥ 128 is dropped (keeps IDs in i8 range for Rust compatibility).

Usage:
    python train.py --data-dir /path/to/text/files [--steps 50000] [--device cuda]

Output:
    out/mini.pt          — final PyTorch checkpoint
    out/weights/         — per-tensor binary weight files (for quantize.py)
"""
import argparse
import math
import os
import time
import struct
import glob

import torch
import torch.optim as optim

from config import MiniConfig
from model import S7MiniModel


# ── Data ──────────────────────────────────────────────────────────────────────

class ByteDataset(torch.utils.data.Dataset):
    """
    Memory-mapped byte-level dataset from a flat binary file.

    Binary file layout: raw uint8 bytes (one per ASCII character, ≥128 dropped).
    Build with: python train.py --build-data --data-dir <dir> --out data.bin
    """
    def __init__(self, path: str, seq_len: int):
        import numpy as np
        self.seq_len = seq_len
        self.data    = np.memmap(path, dtype=np.uint8, mode="r")
        self.n       = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        start  = idx * self.seq_len
        chunk  = self.data[start : start + self.seq_len + 1].astype("int64")
        tokens  = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:])
        return tokens, targets


def build_data_bin(data_dir: str, out_path: str):
    """Encode text files as ASCII bytes and write a flat uint8 binary."""
    import numpy as np

    paths  = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
    paths += sorted(glob.glob(os.path.join(data_dir, "**/*.md"),  recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .txt or .md files found under {data_dir}")

    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        # Keep only ASCII bytes 1-127; map 0 to 1 (reserve 0 for EOS/UNK).
        out.extend(b for b in text.encode("ascii", errors="replace") if 1 <= b < 128)

    arr = np.array(out, dtype=np.uint8)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr.tofile(out_path)
    print(f"[data] wrote {len(arr):,} bytes → {out_path}")
    return out_path


# ── LR Schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, cfg: MiniConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: S7MiniModel, step: int, loss: float, cfg: MiniConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"step": step, "loss": loss, "state": model.state_dict()}, cfg.ckpt_path)
    print(f"[ckpt] step={step}  loss={loss:.4f}  → {cfg.ckpt_path}")


def load_checkpoint(model: S7MiniModel, cfg: MiniConfig) -> int:
    if not os.path.exists(cfg.ckpt_path):
        return 0
    ckpt  = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state"])
    step  = ckpt["step"]
    print(f"[ckpt] resumed from step={step}  loss={ckpt['loss']:.4f}")
    return step


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(cfg: MiniConfig, data_path: str, device: torch.device, resume: bool = False):
    model = S7MiniModel(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] S7-LLM-Mini  params={n_params:,}")
    budget = cfg.param_budget()
    for k, v in budget.items():
        print(f"  {k}: {v:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    dataset = ByteDataset(data_path, cfg.max_seq_len)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    data_iter = iter(loader)

    start_step = load_checkpoint(model, cfg) if resume else 0
    model.train()

    t0 = time.time()
    for step in range(start_step, cfg.max_steps):
        # Fetch next batch, cycling the iterator.
        try:
            tokens, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            tokens, targets = next(data_iter)

        tokens  = tokens.to(device)
        targets = targets.to(device)

        # Update learning rate.
        lr = cosine_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        out  = model(tokens, targets)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0:
            dt   = time.time() - t0
            tok  = cfg.batch_size * cfg.max_seq_len * cfg.log_every
            tps  = tok / max(dt, 1e-6)
            print(f"step={step:6d}  loss={loss.item():.4f}  lr={lr:.2e}  tok/s={tps:,.0f}")
            t0 = time.time()

        if step % cfg.ckpt_every == 0 and step > 0:
            save_checkpoint(model, step, loss.item(), cfg)

    save_checkpoint(model, cfg.max_steps, loss.item(), cfg)
    print("[train] done.")
    return model


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train S7-LLM-Mini")
    parser.add_argument("--data-dir",  default=None,        help="directory of .txt/.md files")
    parser.add_argument("--data-bin",  default="out/data.bin", help="path to pre-built data binary")
    parser.add_argument("--build-data", action="store_true", help="encode data-dir → data-bin then train")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",    type=int,  default=None)
    parser.add_argument("--batch",    type=int,  default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    cfg    = MiniConfig()
    device = torch.device(args.device)

    if args.steps is not None: cfg.max_steps    = args.steps
    if args.batch is not None: cfg.batch_size   = args.batch
    if args.lr    is not None: cfg.learning_rate = args.lr

    data_bin = args.data_bin
    if args.build_data:
        if not args.data_dir:
            parser.error("--build-data requires --data-dir")
        data_bin = build_data_bin(args.data_dir, data_bin)
    elif not os.path.exists(data_bin):
        if args.data_dir:
            data_bin = build_data_bin(args.data_dir, data_bin)
        else:
            parser.error(f"data binary not found: {data_bin}  (pass --data-dir to build it)")

    train(cfg, data_bin, device, resume=args.resume)


if __name__ == "__main__":
    main()
