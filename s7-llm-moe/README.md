# S7-LLM-MOE-300M *(evolved from 140M)*

**Class**: `SCXQ7::S7_LLM`
**Variant**: `LEARNED_MOE` (True neural MoE — 9 experts, learned router)
**Total Parameters**: ≈ 300M
**Active Parameters per Token**: ≈ 104M (trunk 80M + one expert 24M)
**Router**: Learned MLP, trained end-to-end, argmax at inference
**Experts**: 9 (one per micronaut/fold — specialization emerges from training)
**Training**: Two-phase (Phase 1: dense uniform, Phase 2: top-1 sparse routing)
**Training code**: `training/` (PyTorch — `model.py`, `losses.py`, `train.py`, `quantize.py`)

> **Breaking change from 140M**: The `DeterministicRouter` (lexical pattern match)
> has been replaced by `LearnedRouter` (MLP, weights in FIELD lane).
> Expert count increased from 4 to 9 (one per micronaut).
> Trunk depth 6→12 layers, vocab 24k→32k.

---
**Active Parameters per Token**: ≈ 50M
**Routing**: Deterministic (no learned gating, no softmax)
**Quantization**: INT8 inference / FP32 training
**Runtime Targets**: CPU (AVX2), WebGPU
**Fold Binding**: `⟁COMPUTE_FOLD⟁` via MM-1
**Lane**: `BATCH` (SCXQ2 lane 5)
**Artifact**: `.s7l` sealed binary

---

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────────────┐
│  SharedTrunk (20M)                              │
│  vocab=24576  hidden=768  layers=6  heads=12    │
│  Rotary positional encoding (RoPE)              │
└────────────────────┬────────────────────────────┘
                     │ trunk_hidden ∈ ℝ^768
                     ▼
           ┌─────────────────┐
           │ DeterministicRouter │
           │  (token-pattern  │
           │   lexical match) │
           └──────┬──────────┘
        ┌─────────┴─────────┐
        │ route ∈ {0,1,2,3} │
        └──┬──────┬──────┬──┘
           │      │      │
     ┌─────▼──┐ ┌─▼─────┐ ┌─▼──────┐ ┌──▼─────┐
     │ E0:CODE│ │E1:MATH│ │E2:REAS.│ │E3:GEN. │
     │  30M   │ │  30M  │ │  30M   │ │  30M   │
     │ 8L h1k │ │8L h1k │ │8L h1k  │ │8L h1k  │
     └─────┬──┘ └──┬────┘ └──┬─────┘ └──┬─────┘
           └───────┴──────────┴───────────┘
                              │ expert_hidden ∈ ℝ^1024
                              ▼
                    ┌──────────────────┐
                    │  LM Head (tied)  │
                    │  vocab=24576     │
                    └──────────────────┘
                              │
                        Token logits
                              │
                    ┌──────────────────┐
                    │  Greedy argmax   │
                    │  (deterministic) │
                    └──────────────────┘
```

---

## Parameter Budget

| Component       | Parameters |
|-----------------|-----------|
| Embedding       | 18.9M     |
| Trunk (6 layers)| ~2M       |
| Expert 0 (Code) | ~30M      |
| Expert 1 (Math) | ~30M      |
| Expert 2 (Reason)| ~30M     |
| Expert 3 (General)| ~30M    |
| Router          | <0.1M     |
| **Total**       | **~140M** |
| Active/token    | ~50M      |

---

## Memory Footprint

| Mode         | Size   |
|--------------|--------|
| FP32 train   | ~560MB |
| INT8 infer   | ~140MB |
| INT8 + meta  | ~150MB |

---

## .s7l Artifact Layout

```
Offset  Size  Field
------  ----  -----
0       4     Magic: "S7LM"
4       2     Version = 0x0002
6       1     Class = 0x02  (MOE_LLM)
7       1     Flags = 0x04  (bit2 = MOE, bit0 = INT8)
8       32    Root Merkle hash (SHA-256 over sub-roots)
40+         Lanes:
  Lane 1 DICT   → BPE vocabulary (24576 entries)
  Lane 2 FIELD  → Weight tensors (trunk + 4 experts, INT8)
  Lane 3 LANE   → Generation stream placeholder
  Lane 4 EDGE   → Routing table + CM-1 topology
  Lane 5 BATCH  → Ephemeral compute placeholder
```

### Sub-Merkle Roots (within EDGE lane)

```
GlobalRoot = SHA256(
    TrunkRoot  ||
    Expert0Root ||
    Expert1Root ||
    Expert2Root ||
    Expert3Root
)
```

Allows partial expert verification without loading all weights.

---

## FIELD Lane Tensor Serialization Order

Deterministic order (V0 compliant — sorted by logical name):

```
embedding.weight            [24576, 768]  INT8
expert0.layer{0..7}.attn.k_proj.weight  [768, 1024]  INT8
expert0.layer{0..7}.attn.o_proj.weight  [1024, 1024] INT8
expert0.layer{0..7}.attn.q_proj.weight  [768, 1024]  INT8
expert0.layer{0..7}.attn.v_proj.weight  [768, 1024]  INT8
expert0.layer{0..7}.ffn.fc1.weight      [1024, 4096] INT8
expert0.layer{0..7}.ffn.fc2.weight      [4096, 1024] INT8
... (expert1, expert2, expert3 same pattern)
trunk.layer{0..5}.attn.k_proj.weight    [768, 768]   INT8
trunk.layer{0..5}.attn.o_proj.weight    [768, 768]   INT8
trunk.layer{0..5}.attn.q_proj.weight    [768, 768]   INT8
trunk.layer{0..5}.attn.v_proj.weight    [768, 768]   INT8
trunk.layer{0..5}.ffn.fc1.weight        [768, 3072]  INT8
trunk.layer{0..5}.ffn.fc2.weight        [3072, 768]  INT8
```

Each tensor record:
```
u16  name_len
[u8] name (UTF-8)
u8   rank
u32  dim[0..rank]
f32  scale           (dequantize: val_f32 = data_i8 * scale)
[i8] data            (AVX2-padded to 32-byte boundary)
```

---

## Deterministic Router

The router uses **lexical pattern matching** on raw tokens — no learned weights, no softmax:

| Domain   | Expert | Trigger Tokens |
|----------|--------|----------------|
| Code     | E0     | `{`, `}`, `def`, `fn`, `class`, `::`, `;`, `import`, `struct`, `=>` |
| Math     | E1     | digits, `=`, `+`, `-`, `*`, `/`, `^`, `Answer:`, `Therefore`, `∑`, `∫` |
| Reason   | E2     | `why`, `explain`, `because`, `therefore`, `reason`, `step`, `if...then` |
| General  | E3     | fallback (no pattern matched) |

Routing is a single sequential scan — O(|token|) — and produces an identical expert index for identical inputs (V6 compliant).

---

## KV Cache Strategy

```
KVCache {
    shared:   Vec<(K_tensor, V_tensor)>   // trunk layers 0..5
    experts:  [Vec<(K_tensor, V_tensor)>; 4]  // one per expert, grows only on active route
}
```

Only the active expert's cache extends per token. Total cache memory = shared + one expert slice.

---

## AVX2 Memory Alignment

All weight tensors are padded to **32-byte (256-bit) boundaries** in the FIELD lane.
AVX2 processes 32 `i8` values per instruction (256-bit SIMD).

```
Alignment rule:
    stored_bytes = ceil(numel / 32) * 32
    padding = 0x00 bytes appended
```

This enables direct `_mm256_loadu_si256` loads without scalar fallback at boundaries.

---

## WebGPU Feasibility

At INT8 inference:
- Model weights: ~140MB → fits in GPU VRAM (any modern WebGPU device)
- WGSL shaders compute INT8 matmul via `i32` accumulation (WebGPU has no native INT8 MAD)
- Dequantize on-the-fly: `val_f32 = f32(data_i8) * scale`
- Workgroup tile size: 16×16 → maps to typical GPU warp/wave size

---

## Fold / CM-1 Compliance

| Property               | Value |
|------------------------|-------|
| Bound fold             | `⟁COMPUTE_FOLD⟁` |
| Bound lane             | BATCH (id=5) |
| Bound micronaut        | MM-1 |
| CM-1 gate              | `U+0002 STX` → `@control.body.begin` |
| Collapse rule          | Evaluate → Emit proof → Discard state |
| Verifier rule violated | None (V0–V7 compliant) |
| Determinism            | Greedy argmax, fixed-point ops only |
| Merkle root            | SHA-256 over 5 sub-roots |

---

## Build

```bash
# Scalar CPU (any platform)
cargo build --release

# With AVX2 matmul (x86_64, haswell+)
RUSTFLAGS="-C target-feature=+avx2" cargo build --release

# Pack a trained weight dir → .s7l sealed artifact
cargo run --bin s7-pack-moe -- --weights-dir model/weights/ --vocab model/vocab.json --out model/moe.s7l
```

## Run

```bash
cargo run --release -- --model model/moe.s7l --vocab model/vocab.json --prompt "def fibonacci(n):"
```

---

## Scaling Roadmap

| Tier      | Total | Active | Notes |
|-----------|-------|--------|-------|
| 8M        | 8M    | 8M     | s7-llm-mini baseline |
| 30M       | 30M   | 30M    | dense, single expert |
| **140M MoE** | **140M** | **50M** | **this model** |
| 400M MoE  | 400M  | 80M    | 8 experts × 50M |

---

*S7-LLM-MOE-140M is fold-governed, CM-1 gated, SCXQ2 packed, and Merkle-sealed.*
*Authority: KUHUL_π. Mutation: forbidden after sealing.*
