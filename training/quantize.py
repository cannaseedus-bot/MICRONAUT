"""
S7-LLM-MOE-300M Post-Training Quantization and Sealing.

Pipeline:
    1. Load FP32/BF16 checkpoint.
    2. Run INT8 quantization (per-row absmax scaling).
    3. Write weight .bin files + scales.json (for s7-pack-moe).
    4. Verify: reload and compare FP32 vs INT8 model logits.
    5. Call s7-pack-moe to seal into .s7l artifact.

Usage:
    python quantize.py \
        --checkpoint checkpoints/moe-300m/phase2_final/checkpoint.pt \
        --vocab      model/vocab.json \
        --out-dir    model/weights/ \
        --calibrate  data/tokenized/calib.bin \
        --seal       model/moe-300m.s7l

INT8 quantization spec:
    For each weight matrix W [out, in]:
        scale[j] = max(|W[j, :]|) / 127.0      (per-output-row absmax)
        W_q[j, i] = round(W[j, i] / scale[j]).clamp(-128, 127)

    Dequantize at inference: W_f32[j, i] = W_q[j, i] * scale[j]

AVX2 padding:
    Each row is padded to a 32-byte boundary (32 i8 = one AVX2 register).
    Padding bytes are 0x00 and do not affect computation.

Tensor serialization order (FIELD lane, V0 deterministic):
    Sorted lexicographically by tensor name.
"""
import os
import json
import math
import struct
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from model import S7LlmMoe300M
from config import (
    VOCAB_SIZE, TRUNK_HIDDEN, TRUNK_LAYERS, EXPERT_HIDDEN,
    EXPERT_LAYERS, NUM_EXPERTS, EXPERT_FFN_DIM, TRUNK_FFN_DIM,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AVX2_ALIGN = 32  # bytes (256-bit SIMD register width in i8 lanes)


# ── Quantisation ──────────────────────────────────────────────────────────────

def quantize_tensor(w: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row absmax INT8 quantization.

    w: [out_dim, in_dim] FP32/BF16
    Returns:
        w_q:    [out_dim, in_dim_padded] np.int8  (AVX2-padded)
        scales: [out_dim]                np.float32
    """
    w_fp32 = w.float()
    out_dim, in_dim = w_fp32.shape

    # Per-row scale: absmax / 127.
    scales = w_fp32.abs().max(dim=1).values / 127.0
    scales = scales.clamp(min=1e-8)   # avoid division by zero

    # Quantize.
    w_scaled = w_fp32 / scales.unsqueeze(1)           # [out, in]
    w_q = w_scaled.round().clamp(-128, 127).to(torch.int8)

    # AVX2 padding: pad in_dim to multiple of 32.
    pad_in = math.ceil(in_dim / AVX2_ALIGN) * AVX2_ALIGN
    if pad_in > in_dim:
        pad = torch.zeros(out_dim, pad_in - in_dim, dtype=torch.int8)
        w_q = torch.cat([w_q, pad], dim=1)

    return w_q.numpy(), scales.numpy()


# ── Calibration-Based Scale Refinement ───────────────────────────────────────

def calibrate_scales(
    model: S7LlmMoe300M,
    calib_path: str,
    seq_len: int = 512,
    n_samples: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Run calibration samples through the model to collect activation ranges.
    Returns a dict of tensor_name → refined_scale (for output projection layers).

    For weight-only quantization (our case) this step is optional;
    per-row absmax is already a strong baseline.
    We include this for completeness and future activation quantization.
    """
    log.info("Running calibration with %d samples from %s", n_samples, calib_path)
    data = np.memmap(calib_path, dtype=np.uint16, mode="r")
    model.eval()
    model = model.to(device)

    activation_maxima: Dict[str, float] = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                v = out.float().abs().max().item()
                activation_maxima[name] = max(activation_maxima.get(name, 0.0), v)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    stride = seq_len + 1
    n_avail = len(data) // stride
    idxs = np.random.choice(n_avail, size=min(n_samples, n_avail), replace=False)

    with torch.no_grad():
        for idx in idxs:
            row = torch.from_numpy(
                data[idx * stride : idx * stride + seq_len].astype(np.int64)
            ).unsqueeze(0).to(device)
            model(row)

    for h in hooks:
        h.remove()

    log.info("Calibration complete. %d activation ranges recorded.", len(activation_maxima))
    return activation_maxima


# ── Weight Export ─────────────────────────────────────────────────────────────

def extract_state_dict(model: S7LlmMoe300M) -> Dict[str, torch.Tensor]:
    """
    Return a flat dict of (canonical_name → tensor) for all weight matrices.
    Names match the FIELD lane serialization order expected by s7-pack-moe.
    """
    state = {}
    sd = model.state_dict()

    for key, tensor in sd.items():
        if tensor.dim() < 2:
            continue  # skip biases and 1D norms
        state[key] = tensor

    return state


def write_weights(
    model: S7LlmMoe300M,
    out_dir: str,
) -> Dict[str, float]:
    """
    Quantize all weight tensors and write:
        {out_dir}/{tensor_name}.bin   — INT8 data, AVX2-padded, row-major
        {out_dir}/scales.json         — {tensor_name: scale_f32}

    Returns scales dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    weights = extract_state_dict(model)
    all_scales = {}

    log.info("Quantizing %d weight tensors...", len(weights))

    for name, tensor in sorted(weights.items()):
        log.info("  %-60s %s", name, list(tensor.shape))
        w_q, scales = quantize_tensor(tensor)

        # Write INT8 binary.
        fname = name.replace("/", ".") + ".bin"
        fpath = os.path.join(out_dir, fname)
        w_q.tofile(fpath)

        # Store per-row scales; use mean scale as single float for simple lookup.
        # s7-pack-moe reads scales.json as {name: float32_scale}.
        all_scales[fname.replace(".bin", "")] = float(scales.mean())

    scales_path = os.path.join(out_dir, "scales.json")
    with open(scales_path, "w") as f:
        json.dump(all_scales, f, indent=2, sort_keys=True)

    log.info("Weights written to %s (%d files + scales.json)", out_dir, len(weights))
    return all_scales


# ── Quantization Quality Validation ──────────────────────────────────────────

@torch.no_grad()
def validate_quantization(
    model_fp32: S7LlmMoe300M,
    weights_dir: str,
    n_tokens: int = 128,
    max_kl_div: float = 0.05,
) -> bool:
    """
    Quick sanity check: compare logit distributions of FP32 vs INT8 model.
    Passes if mean KL divergence < max_kl_div per token.
    """
    log.info("Validating quantization quality...")
    model_fp32.eval()

    dummy_tokens = torch.randint(0, 32768, (1, n_tokens))
    out_fp32     = model_fp32(dummy_tokens)
    probs_fp32   = torch.softmax(out_fp32["logits"][0], dim=-1)

    # Reload model with quantized weights (dequantized in-place for comparison).
    model_q = S7LlmMoe300M()
    model_q.eval()

    state = model_fp32.state_dict()
    for name, tensor in list(state.items()):
        if tensor.dim() < 2:
            continue
        fname = os.path.join(weights_dir, name.replace("/", ".") + ".bin")
        if not os.path.exists(fname):
            continue
        out_dim, in_dim = tensor.shape[:2]
        data = np.fromfile(fname, dtype=np.int8)
        pad_in = math.ceil(in_dim / AVX2_ALIGN) * AVX2_ALIGN
        w_q = torch.from_numpy(data.reshape(out_dim, pad_in)[:, :in_dim])
        scale_mean = tensor.float().abs().max() / 127.0
        state[name] = w_q.float() * scale_mean.item()

    model_q.load_state_dict(state, strict=False)
    out_q     = model_q(dummy_tokens)
    probs_q   = torch.softmax(out_q["logits"][0], dim=-1)

    kl_per_token = torch.nn.functional.kl_div(
        probs_q.log().clamp(min=-100),
        probs_fp32,
        reduction="batchmean",
    ).item()

    log.info("Mean KL divergence FP32 vs INT8: %.6f (threshold: %.3f)", kl_per_token, max_kl_div)
    passed = kl_per_token < max_kl_div
    if passed:
        log.info("✓ Quantization quality: PASS")
    else:
        log.warning("✗ Quantization quality: FAIL (KL=%.4f > %.3f)", kl_per_token, max_kl_div)
    return passed


# ── Sealing (calls s7-pack-moe) ───────────────────────────────────────────────

def seal_artifact(
    weights_dir: str,
    vocab_path:  str,
    out_path:    str,
    pack_bin:    str = "s7-llm-moe/target/release/s7-pack-moe",
):
    """
    Call the Rust s7-pack-moe binary to seal the weights into a .s7l artifact.
    """
    if not os.path.exists(pack_bin):
        raise FileNotFoundError(
            f"s7-pack-moe binary not found: {pack_bin}\n"
            "Build with: cargo build --release --manifest-path s7-llm-moe/Cargo.toml"
        )

    cmd = [
        pack_bin,
        "--weights-dir", weights_dir,
        "--vocab",       vocab_path,
        "--out",         out_path,
    ]
    log.info("Sealing artifact: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"s7-pack-moe failed:\n{result.stderr}")

    log.info("Sealed artifact: %s", out_path)
    log.info("%s", result.stderr.strip())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to FP32/BF16 checkpoint.pt")
    parser.add_argument("--vocab",       default="model/vocab.json")
    parser.add_argument("--out-dir",     default="model/weights/",
                        help="Directory to write INT8 weight .bin files")
    parser.add_argument("--seal",        default=None,
                        help="Output .s7l path (if set, calls s7-pack-moe)")
    parser.add_argument("--calibrate",   default=None,
                        help="Optional calibration .bin path")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--pack-bin",    default="s7-llm-moe/target/release/s7-pack-moe")
    args = parser.parse_args()

    # Load model.
    log.info("Loading checkpoint: %s", args.checkpoint)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = S7LlmMoe300M()
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Model loaded. Parameters: %dM", model.param_count() // 1_000_000)

    # Calibration (optional).
    if args.calibrate:
        calibrate_scales(model, args.calibrate)

    # Quantize and write weights.
    write_weights(model, args.out_dir)

    # Validation.
    if not args.skip_validate:
        passed = validate_quantization(model, args.out_dir)
        if not passed:
            log.warning("Quantization validation failed — inspect KL divergence before sealing.")

    # Seal.
    if args.seal:
        seal_artifact(args.out_dir, args.vocab, args.seal, args.pack_bin)
        log.info("Done: %s", args.seal)
    else:
        log.info("Done. Use --seal <out.s7l> to pack into a sealed artifact.")


if __name__ == "__main__":
    main()
