# S7-LLM-CODE-8M Family Specification

## Identity

```text
Class: SCXQ7::S7_LLM
Tier: SMALL
Variant: CODE
Param Budget: ~8,000,000
Quantization: INT8 (inference)
Training: FP32 deterministic
```

S7-LLM-CODE-8M is the first serious small-model tier in the fold-governed S7 line, designed as a **family** of deterministic variants rather than a single checkpoint.

## 1) Architecture Upgrade (8M Tier)

### Initial target envelope

| Field          | Value       |
| -------------- | ----------- |
| Vocab          | 4096        |
| Context        | 512         |
| Hidden         | 384         |
| Layers         | 8           |
| Heads          | 6           |
| Head Dim       | 64          |
| FFN Multiplier | 4x          |
| Rotary         | Yes         |
| Dropout        | 0           |
| Attention      | Full causal |

### Rough parameter estimate

- Embedding: `4096 × 384 ≈ 1.57M`
- Per-layer attention:
  - `QKV: 384 × (3×384)`
  - `Wo: 384 × 384`
- Per-layer FFN:
  - `384 × 1536`
  - `1536 × 384`
- Raw 8-layer stack overshoots target unless compressed.

### Balanced spec near 8M

| Field    | Value |
| -------- | ----- |
| Hidden   | 320   |
| Layers   | 8     |
| Heads    | 5     |
| Head Dim | 64    |
| Vocab    | 4096  |

This balanced profile lands in the practical 7–9M band with:

- embedding/lm_head tying,
- compact gated FFN,
- projection sharing where permitted by fold constraints.

## 2) Family Variants (SKUs)

### A) `S7-LLM-CODE-8M`

Primary domains:

- Python, JavaScript, Bash, Rust, C-like code generation and repair.

Training mix:

- 40% permissive code,
- 25% instruction format,
- 20% math reasoning,
- 15% natural language.

### B) `S7-LLM-MATH-8M`

Primary domains:

- GSM8K-style arithmetic,
- scratchpad steps,
- proof-like decomposition.

Reserved structure tokens:

- `<REASON>`
- `<THINK>`
- `<FINAL>`

### C) `S7-LLM-GENERAL-8M`

Balanced multi-domain checkpoint with moderate coding and reasoning coverage.

### D) `S7-LLM-MOE-8M` (Experimental)

Deterministic two-expert routing:

- Expert 1: CODE
- Expert 2: GENERAL/MATH

Reference deterministic router:

```python
if token contains "{", "def", ";", "fn":
    route CODE
else:
    route GENERAL
```

No probabilistic dispatch, no stochastic top-k gating.

## 3) Deterministic Training Strategy

To preserve fold-law consistency and run-to-run stability:

- fixed dataset order,
- fixed domain interleave,
- no shuffle,
- no dropout,
- Adam with fixed seed,
- no AMP,
- no gradient noise.

## 4) Dataset Strategy (Legal + Effective)

Preferred sources (license verified):

- permissive GitHub repos (MIT/Apache/BSD),
- permissive The Stack subsets,
- HumanEval (MIT),
- CodeAlpaca where license chain is clear,
- Evol-Instruct-Code where licensing is auditable,
- synthetic template expansions.

## 5) Structured Synthetic Curriculum

Small models improve significantly with dense formatting priors.

Reference prompt format:

```text
### Instruction:
Write a function to reverse a string.

### Response:
def reverse_string(s):
    return s[::-1]
```

Curriculum should include:

- bug explanations,
- repair diffs,
- refactor tasks,
- unit-test generation.

## 6) S7-LLM-CODE-8M Lane Layout

| Lane  | Content                |
| ----- | ---------------------- |
| DICT  | 4096 BPE vocab         |
| EDGE  | Transformer graph      |
| FIELD | FP32 training weights  |
| FIELD | INT8 quantized weights |
| LANE  | Training trace         |
| EDGE  | CM-1 control           |
| BATCH | Gradient trace hashes  |

## 7) Inference Upgrades

Deterministic runtime profile:

- rotary embeddings,
- KV cache retained in FIELD lane structures,
- greedy decode only (`top-k=1`).

## 8) WebGPU Path (8M Practical Tier)

Deterministic acceleration target:

- `INT8 × INT8 → INT32` kernels,
- tiled matrix multiply,
- shared-memory staging,
- fixed workgroup geometry,
- no atomics in accumulation paths.

## 9) Expected Capability Band

At 8M scale, expected strengths:

- simple function synthesis,
- loops/conditionals,
- small refactors,
- lightweight GSM8K arithmetic,
- stable instruction formatting.

Expected limits:

- weak long-horizon system construction,
- limited deep multi-hop reasoning,
- below 100M+ class model robustness.

## 10) Forward Scaling Plan

| Tier | Params      | Use Case         |
| ---- | ----------- | ---------------- |
| 1M   | Demo        | Law proof        |
| 8M   | Coding mini | Embedded         |
| 30M  | Real coding | Desktop          |
| 100M | Strong      | Edge workstation |
| 300M | Production  | Serious          |

## Compact Identity String

```text
Fold-governed
CM1-gated
SCXQ2-packed
Merkle-sealed
INT8 deterministic
8M parameter transformer
with structured coding curriculum
```

## 11) Engineering Notes: Deterministic MoE Capacity

MoE experts at this scale do **not** need to be 30M each.

A practical 8M-class MoE split is:

- 4M shared trunk,
- 2M code expert,
- 2M math/general expert.

With deterministic single-expert activation, effective capacity exceeds dense 8M while compute stays near dense cost.

## 12) FIELD Lane Tensor Contract (Exact Packing)

FIELD lane (`id=0x02`) stores repeated packed tensors:

```text
[u16 name_len][name bytes][u8 rank][u32 dims[rank]][f32 scale][i8 data[numel]]
```

Parser requirements:

- reject duplicate tensor names,
- reject rank-0 tensors,
- reject zero dimensions,
- reject truncated records,
- validate `data.len() == product(dims)`.

Required minimum tensors for mini wiring:

- `embedding.weight`
- `lm_head.weight`

## 13) Runtime Wiring (`from_s7`) Behavior

`S7Mini::from_s7()` now follows deterministic behavior:

1. If FIELD lane exists, parse and validate packed tensors, then wire `embedding.weight` and `lm_head.weight`.
2. If FIELD lane is absent (header-only placeholder artifact), bootstrap deterministic zero tensors so the binary remains runnable.

This keeps sealed-lane validation strict while preserving a deterministic fallback for the current minimal artifact.
