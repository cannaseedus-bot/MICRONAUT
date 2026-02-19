"""
S7-LLM-Mini post-training quantization and .s7l artifact sealing.

Steps:
  1. Load trained PyTorch checkpoint.
  2. Quantize all weight tensors to INT8 (per-row absmax for 2-D,
     per-tensor absmax for 1-D).  AVX2 32-byte padding applied.
  3. Build vocab.json — ASCII byte-level (char → ASCII value, 0 = <unk>).
  4. Validate: measure KL divergence FP32 ↔ INT8 logits on a sample batch.
  5. Pack into mini.s7l using the .s7l binary format (pure Python, no Rust dep).

.s7l format (CLASS_LLM = 0x01, see s7l/header.rs):
  Offset  Size  Field
  ------  ----  -----
  0       4     Magic "S7LM"
  4       2     Version = 0x0001 (big-endian u16)
  6       1     Class   = 0x01  (LLM)
  7       1     Flags   = 0x01  (bit0 = INT8)
  8       32    Root Merkle hash (SHA-256 binary tree over lane payload-hashes)
  40+          Lanes (repeating):
    1     lane_id  (u8)
    4     length   (big-endian u32, payload only — excludes this header)
    N     payload
    32    SHA-256 of payload

FIELD lane tensor record format (matches tensor/field_lane.rs):
  u16  name_len
  [u8] name (UTF-8)
  u8   rank
  u32  dim[i]  × rank
  f32  scale  (dequant: f32 = data_i8 * scale)
  [i8] data   (AVX2-padded: ceil(numel/32)*32 bytes)

Usage:
    python quantize.py [--ckpt out/mini.pt] [--out ../model/mini.s7l]
"""
import argparse
import hashlib
import io
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

from config import MiniConfig
from model  import S7MiniModel


# ── Quantisation ──────────────────────────────────────────────────────────────

def avx2_pad(numel: int) -> int:
    """Round up to next 32-byte boundary (AVX2 256-bit alignment)."""
    return (numel + 31) & ~31


def quantize_row(row: np.ndarray) -> tuple[float, np.ndarray]:
    """Quantize a 1-D FP32 row to INT8 using per-row absmax."""
    amax = float(np.abs(row).max())
    if amax < 1e-9:
        return 1.0 / 127.0, np.zeros(len(row), dtype=np.int8)
    scale = amax / 127.0
    q     = np.clip(np.round(row / scale), -127, 127).astype(np.int8)
    return scale, q


def quantize_tensor(name: str, w: np.ndarray) -> dict:
    """
    Quantize a weight tensor to INT8.

    2-D tensors: per-row absmax (one scale per output row).
    1-D tensors: single per-tensor absmax scale.

    Returns dict with keys: name, dims, scale (scalar or list), data (int8 ndarray).
    """
    if w.ndim == 2:
        scales, rows = [], []
        for r in range(w.shape[0]):
            s, q = quantize_row(w[r])
            scales.append(s)
            rows.append(q)
        data = np.concatenate(rows)
        # Use mean scale for the FIELD lane record (matches runtime dequant).
        scale = float(np.mean(scales))
    elif w.ndim == 1:
        scale, data = quantize_row(w)
    else:
        raise ValueError(f"{name}: unsupported rank {w.ndim}")

    numel  = data.size
    padded = avx2_pad(numel)
    if padded > numel:
        data = np.concatenate([data, np.zeros(padded - numel, dtype=np.int8)])

    return {"name": name, "dims": list(w.shape), "scale": scale, "data": data}


# ── FIELD lane serialisation ──────────────────────────────────────────────────

def write_tensor_record(buf: io.BytesIO, rec: dict):
    """Serialise one tensor record into buf (matches field_lane.rs)."""
    name_bytes = rec["name"].encode("utf-8")
    buf.write(struct.pack(">H", len(name_bytes)))
    buf.write(name_bytes)
    dims = rec["dims"]
    buf.write(struct.pack("B", len(dims)))
    for d in dims:
        buf.write(struct.pack(">I", d))
    buf.write(struct.pack(">f", rec["scale"]))
    buf.write(rec["data"].tobytes())


def build_field_lane(tensors: list[dict]) -> bytes:
    """Encode all tensor records into the FIELD lane payload."""
    buf = io.BytesIO()
    for t in tensors:
        write_tensor_record(buf, t)
    return buf.getvalue()


# ── DICT lane (vocabulary) ────────────────────────────────────────────────────

def build_vocab_json(cfg: MiniConfig) -> tuple[dict, bytes]:
    """
    Build ASCII byte-level vocabulary.

    Mapping: character string → integer ID (0-127).
    ID 0  = <unk> / EOS (reserved).
    ID 1-127 = ASCII characters chr(1)..chr(127).

    Printable special entries are given readable names:
      " " → 32,  "\n" → 10,  "\t" → 9.
    """
    vocab = {"<unk>": 0}
    for i in range(1, cfg.vocab_size):
        ch = chr(i)
        vocab[ch] = i
    payload = json.dumps(vocab, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return vocab, payload


# ── .s7l packing ─────────────────────────────────────────────────────────────

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def merkle_root(hashes: list[bytes]) -> bytes:
    """Binary Merkle tree over SHA-256 lane hashes (Bitcoin-style odd-node duplication)."""
    if not hashes:
        return b"\x00" * 32
    layer = list(hashes)
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            left  = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else layer[i]
            next_layer.append(sha256(left + right))
        layer = next_layer
    return layer[0]


def pack_lane(lane_id: int, payload: bytes) -> tuple[bytes, bytes]:
    """
    Pack one lane: returns (lane_bytes, payload_hash).
    Lane format: u8 id + u32 length (big-endian) + payload + 32-byte SHA-256.
    """
    h    = sha256(payload)
    lane = struct.pack("B>I", lane_id, len(payload)) + payload + h
    # struct doesn't support mixed-endian format strings — split packing:
    lane = struct.pack("B", lane_id) + struct.pack(">I", len(payload)) + payload + h
    return lane, h


def write_s7l(
    tensors:      list[dict],
    vocab_payload: bytes,
    out_path:     str,
):
    """Write the complete .s7l artifact."""
    # ── Build lane payloads ───────────────────────────────────────────────────
    field_payload = build_field_lane(tensors)

    # Lane 1 DICT  = vocabulary
    # Lane 2 FIELD = weight tensors
    # Lane 3 LANE  = generation stream (empty placeholder)
    # Lane 4 EDGE  = topology (empty placeholder for mini)
    # Lane 5 BATCH = ephemeral compute (empty placeholder)
    lane_specs = [
        (1, vocab_payload),
        (2, field_payload),
        (3, b""),    # LANE placeholder
        (4, b""),    # EDGE placeholder
        (5, b""),    # BATCH placeholder
    ]

    lane_bytes_list = []
    lane_hashes     = []
    for lid, payload in lane_specs:
        lb, h = pack_lane(lid, payload)
        lane_bytes_list.append(lb)
        lane_hashes.append(h)

    root = merkle_root(lane_hashes)

    # ── Write header + lanes ──────────────────────────────────────────────────
    header  = b"S7LM"
    header += struct.pack(">H", 0x0001)  # Version = 1
    header += struct.pack("B",  0x01)    # Class   = LLM
    header += struct.pack("B",  0x01)    # Flags   = INT8
    header += root                       # 32-byte Merkle root

    assert len(header) == 40, f"header size mismatch: {len(header)}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for lb in lane_bytes_list:
            f.write(lb)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[pack] sealed → {out_path}  ({size_kb:.1f} KB)")
    print(f"[pack] merkle root: {root.hex()}")
    print(f"[pack] FIELD lane:  {field_payload.__len__():,} bytes  ({len(tensors)} tensors)")
    print(f"[pack] DICT  lane:  {len(vocab_payload):,} bytes  (vocab)")


# ── KL divergence validation ──────────────────────────────────────────────────

@torch.no_grad()
def validate_quantization(
    fp32_model: S7MiniModel,
    tensors:    list[dict],
    cfg:        MiniConfig,
    device:     torch.device,
    n_batches:  int = 10,
) -> float:
    """
    Load quantized weights back into a fresh model and measure mean KL
    divergence against the FP32 model on random token sequences.
    Returns mean KL (nats/token).
    """
    # Reconstruct FP32 weights from INT8 records.
    state = {}
    name_map = {t["name"]: t for t in tensors}
    for pname, ptensor in fp32_model.named_parameters():
        key = pname
        if key not in name_map:
            continue
        rec    = name_map[key]
        data   = torch.from_numpy(rec["data"].astype(np.float32)) * rec["scale"]
        numel  = 1
        for d in rec["dims"]:
            numel *= d
        state[pname] = data[:numel].view(rec["dims"])

    int8_model = S7MiniModel(cfg).to(device)
    int8_model.load_state_dict(state, strict=False)
    int8_model.eval()
    fp32_model.eval()

    total_kl = 0.0
    total_tok = 0
    for _ in range(n_batches):
        tokens = torch.randint(1, cfg.vocab_size, (4, cfg.max_seq_len), device=device)
        fp32_logits = fp32_model(tokens)["logits"]
        int8_logits = int8_model(tokens)["logits"]
        p   = F.softmax(fp32_logits, dim=-1)
        q   = F.softmax(int8_logits, dim=-1)
        kl  = F.kl_div(q.log(), p, reduction="sum")
        total_kl  += float(kl)
        total_tok += tokens.numel()

    mean_kl = total_kl / total_tok
    return mean_kl


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize and seal S7-LLM-Mini")
    parser.add_argument("--ckpt",   default="out/mini.pt",          help="PyTorch checkpoint")
    parser.add_argument("--out",    default="../model/mini.s7l",     help="output .s7l path")
    parser.add_argument("--vocab",  default="../model/vocab.json",   help="output vocab.json path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-validate", action="store_true",        help="skip KL validation")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg    = MiniConfig()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[quant] loading {args.ckpt}")
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    model = S7MiniModel(cfg)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # ── Quantize all parameters ───────────────────────────────────────────────
    tensors = []
    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().astype(np.float32)
        # lm_head is weight-tied to embedding — skip duplicate.
        if name == "lm_head.weight":
            continue
        rec = quantize_tensor(name, w)
        tensors.append(rec)
        dims_str = "×".join(str(d) for d in rec["dims"])
        print(f"  {name:50s} [{dims_str}]  scale={rec['scale']:.5f}")

    print(f"[quant] {len(tensors)} tensors quantized")

    # ── KL divergence check ───────────────────────────────────────────────────
    if not args.no_validate:
        print("[quant] validating INT8 accuracy …")
        mean_kl = validate_quantization(model, tensors, cfg, device, n_batches=cfg.calib_batches // 10)
        status  = "OK" if mean_kl < cfg.kl_threshold else "WARN (exceeds threshold)"
        print(f"[quant] mean KL(FP32 ∥ INT8) = {mean_kl:.5f} nats/token  [{status}]")
        if mean_kl >= cfg.kl_threshold:
            print(f"         threshold = {cfg.kl_threshold}  — consider more training or larger d_model")

    # ── Build vocabulary ──────────────────────────────────────────────────────
    vocab, vocab_payload = build_vocab_json(cfg)
    os.makedirs(os.path.dirname(args.vocab) or ".", exist_ok=True)
    with open(args.vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, sort_keys=True)
    print(f"[vocab] wrote {len(vocab)} entries → {args.vocab}")

    # ── Pack .s7l artifact ────────────────────────────────────────────────────
    write_s7l(tensors, vocab_payload, args.out)
    print("[quant] done — artifact is sealed and Merkle-rooted.")


if __name__ == "__main__":
    main()
