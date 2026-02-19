# S7-LLM-MOE-140M Training Cluster Sizing

**Authority**: KUHUL_π
**Model**: S7-LLM-MOE-140M
**Fold**: ⟁COMPUTE_FOLD⟁
**Training regime**: Two-phase (trunk-first, then domain experts)

---

## 1. Training Architecture Overview

```
Phase 1: Train SharedTrunk on all domains jointly
Phase 2: Freeze trunk → Train each expert on its domain slice
Phase 3: Quantize (FP32 → INT8) → Seal (.s7l) → Verify
```

All phases produce deterministic weight files.
Sealing is irreversible: once packed into .s7l, weights are read-only.

---

## 2. Model Parameter Budget

| Component          | FP32 Params | FP32 Bytes |
|--------------------|-------------|------------|
| Embedding (tied)   | 18.9M       | 75.5MB     |
| Trunk layers (×6)  | ~2M         | ~8MB       |
| Expert 0 (Code)    | ~28M        | ~112MB     |
| Expert 1 (Math)    | ~28M        | ~112MB     |
| Expert 2 (Reason)  | ~28M        | ~112MB     |
| Expert 3 (General) | ~28M        | ~112MB     |
| Router             | <0.1M       | <1MB       |
| **Total FP32**     | **~133M**   | **~532MB** |
| INT8 inference     | —           | ~133MB     |

---

## 3. Phase 1: SharedTrunk Training

### Goal
Train the 6-layer trunk on all four domain corpora simultaneously so that
the trunk captures universal language structure, syntax, and cross-domain
representations.

### Data Mix

| Domain   | Ratio | Source examples                        |
|----------|-------|----------------------------------------|
| Code     | 35%   | The Stack, CodeParrot, GitHub-Code     |
| Math     | 25%   | MathPile, OpenWebMath, AMPS, Minerva   |
| Reasoning| 20%   | OpenHermes, Chain-of-Thought traces    |
| General  | 20%   | RedPajama-v2, Dolma, C4                |

Total tokens for trunk training: **50B tokens** (minimum viable).

### GPU Requirement (Phase 1)

| Parameter          | Value                             |
|--------------------|-----------------------------------|
| Model size (FP32)  | ~83MB (trunk + embedding)         |
| Batch size         | 512 sequences × 2048 tokens       |
| Activation memory  | ~8GB per GPU (BF16 activations)   |
| Gradient memory    | ~83MB × 2 = ~166MB                |
| Optimizer (AdamW)  | ~83MB × 3 = ~250MB                |
| Per-GPU VRAM       | ~12GB minimum (A100-40GB for margin)|
| GPUs required      | 8× A100-40GB (tensor parallel)    |
| Training tokens    | 50B                               |
| Estimated GPU-hrs  | 8 GPUs × ~1500 hr = **12,000 GPU-hr** |

### Cluster Config (Phase 1)

```
Nodes:           4 nodes × 2× A100 (8 GPUs total)
Interconnect:    NVLink within node; 100Gb InfiniBand across nodes
Precision:       BF16 forward + FP32 gradient accumulation
Gradient sync:   AllReduce (NCCL)
Checkpoint step: every 5B tokens
Batch schedule:  cosine decay LR, warmup 2000 steps
```

---

## 4. Phase 2: Expert Training

### Goal
Each expert trains independently on its domain slice only.
Trunk weights are frozen during this phase.
Experts can train in parallel across separate GPU pools.

### Data Per Expert

| Expert | Domain   | Tokens  | Source                          |
|--------|----------|---------|---------------------------------|
| E0     | Code     | 20B     | The Stack v2 (deduplicated)     |
| E1     | Math     | 15B     | MathPile + AMPS + synthetic     |
| E2     | Reason   | 12B     | OpenHermes + CoT + NaturalSteps |
| E3     | General  | 12B     | Dolma + RedPajama               |

### GPU Requirement (Phase 2, per expert)

| Parameter        | Value                              |
|------------------|------------------------------------|
| Expert size(FP32)| ~28M × 4B = ~112MB                |
| Frozen trunk     | ~83MB (loaded, no grad)            |
| Batch size       | 256 × 2048                         |
| Activation mem   | ~4GB per GPU                       |
| Optimizer states | ~112MB × 3 = ~336MB                |
| Per-GPU VRAM     | ~8GB (fits A100-40GB easily)       |
| GPUs per expert  | 4× A100-40GB                       |
| Experts in parallel | 4                               |
| Total GPUs       | **16× A100-40GB**                  |

### Cluster Config (Phase 2)

```
Expert 0 (Code):    Node A → 4× A100, ~600 GPU-hr
Expert 1 (Math):    Node B → 4× A100, ~450 GPU-hr
Expert 2 (Reason):  Node C → 4× A100, ~360 GPU-hr
Expert 3 (General): Node D → 4× A100, ~360 GPU-hr
Total GPU-hr Phase 2: ~1,770 GPU-hr (parallel)
Wall-clock:          max(expert_times) ≈ 150 hr on 4 GPUs
```

---

## 5. Phase 3: Quantisation + Sealing

### INT8 Quantisation

Method: **Post-Training Quantization (PTQ) with per-row scale calibration**.

```
For each weight tensor W[in_dim, out_dim]:
    scale_row[j] = max(|W[:, j]|) / 127.0
    W_int8[:, j] = round(W[:, j] / scale_row[j]).clamp(-128, 127)
```

Calibration data: 4096 representative samples from each domain (held-out).

### Quantisation Quality Target

| Metric         | FP32 | INT8 (target) |
|----------------|------|---------------|
| Code GSM8K     | —    | ≥90% FP32     |
| Math perplexity| —    | ≤+2% FP32     |
| Reason bench   | —    | ≥88% FP32     |

### Sealing Pipeline

```bash
# 1. Quantize weights
python tools/quantize.py \
    --checkpoint checkpoints/trunk_final/ \
    --experts     checkpoints/expert{0,1,2,3}_final/ \
    --out         model/weights/ \
    --method      per_row_int8

# 2. Pack into .s7l artifact
cargo run --bin s7-pack-moe -- \
    --weights-dir model/weights/ \
    --vocab       model/vocab.json \
    --out         model/moe.s7l

# 3. Verify sealed artifact (V0-V7 compliance)
python src/verifier.py model/moe.s7l
```

---

## 6. Total Cluster Budget

| Phase            | GPUs        | GPU-hours | Notes              |
|------------------|-------------|-----------|---------------------|
| Phase 1 (Trunk)  | 8× A100-40G | 12,000    | 50B tokens          |
| Phase 2 (Experts)| 16× A100-40G| 1,770     | 4 experts parallel  |
| Calibration/QA   | 4× A100-40G | 200       | PTQ + validation    |
| **Total**        | **16× peak**| **~14,000**| **~21 days wall-time** |

### Cloud Cost Estimate (A100-40G at $2.50/hr)

```
14,000 GPU-hr × $2.50 = $35,000
```

This is in line with 1B-class model training costs (expected for a 140M dense-equivalent compute budget).

---

## 7. Scaling Invariants

The following invariants must hold at every checkpoint and in the final sealed artifact:

| Rule | Constraint                                           |
|------|------------------------------------------------------|
| V0   | Weight tensors serialised in lexicographic name order|
| V6   | Same input tokens → identical logits (bitwise)       |
| V6   | Same model bytes → identical Merkle root             |
| INT8 | No FP32 ops in inference path (weights, matmul, KV)  |
| CM-1 | STX gate required before any inference call          |
| Fold | ⟁COMPUTE_FOLD⟁ is the sole execution authority       |

---

## 8. AVX2 Expert Layer Memory Alignment

All weight tensors are stored and loaded with **32-byte (256-bit) boundary alignment**.
This enables zero-copy `_mm256_loadu_si256` loads with no scalar tail at boundaries.

### Alignment Rule

```
stored_bytes = ceil(numel / 32) * 32
```

### Per-Expert Layer Memory Layout (Expert, hidden=512)

```
Tensor                  Shape           numel     padded  bytes
─────────────────────── ─────────────── ─────────────────────────
q_proj.weight           [512, 512]      262144    262144  262KB
k_proj.weight           [512, 512]      262144    262144  262KB
v_proj.weight           [512, 512]      262144    262144  262KB
o_proj.weight           [512, 512]      262144    262144  262KB
ffn.fc1.weight          [512, 2048]     1048576   1048576 1024KB
ffn.fc2.weight          [2048, 512]     1048576   1048576 1024KB
─────────────────────── ─────────────── ─────────────────────────
Total per expert layer                              3136KB ≈ 3.1MB
Total 8 layers (1 expert)                          24.8MB
Total 4 experts                                    99.2MB
Trunk layers (6)                       ≈ 24MB
Embedding                              ≈ 18MB
─────────────────────── ─────────────── ─────────────────────────
Grand total (INT8, AVX2-padded)                   ~141MB
```

### Cache Line Alignment (AVX2 → L1/L2)

For optimal L1/L2 cache utilisation during matmul:
- **Tile size**: 32 rows × 32 cols (1KB tile, fits in L1 cache)
- **Prefetch distance**: 2 tiles ahead (hardware prefetch catches most accesses)
- **NUMA locality**: Trunk weights pinned to socket 0; expert weights distributed round-robin across sockets

```
AVX2 register: 256 bits = 32 × i8 = one weight row segment
Tiles per L1 cache (32KB): 32 tiles
Expected L1 hit rate during matmul: >95%
```

---

*This document is the authoritative training sizing specification for S7-LLM-MOE-140M.*
*Authority: KUHUL_π. Fold: ⟁COMPUTE_FOLD⟁. Lane: BATCH.*
