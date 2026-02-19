<img src="logo.svg" width="120">

# MICRONAUT

A **SCO/1 object server** where Micronaut exists as sealed data and file-based projections, with PowerShell acting as the host-native orchestrator.

## What It Is

MICRONAUT treats web primitives as computation infrastructure, not presentation, while enforcing a file-only control plane:

| Primitive | Traditional Use | MICRONAUT Use |
|-----------|-----------------|---------------|
| SVG | Graphics | Spatial state lattice / structured memory |
| CSS | Styling | Deterministic state machine / execution rules |
| PowerShell | Host automation | Orchestrator only (file router, loopback glue) |
| JSON | Configuration | Law / contracts / declarations |

The system is **proof-driven**: the verifier decides correctness, not runtime.

## Architecture

```
Plane 0/1 — Control & Law
├── object.toml            Object server declaration
├── semantics.xjson        KUHUL-TSG schema + CM-1 contract
└── SVG + CSS Micronauts   State lattice + deterministic rule engine

Plane 2 — Sealed Compute
├── brains/                Read-only sealed data
└── micronaut.s7           SCO/1 executable object

Plane 3 — Proof & Replay
├── io/chat.txt            Append-only input (CM-1)
├── io/stream.txt          Append-only semantic emission
├── trace/scxq2.trace       Evidence stream
└── proof/scxq2.proof       Proof artifacts
```

## Directory Structure

```
MICRONAUT/
├── micronaut/                    # SCO/1 object server
│   ├── micronaut.s7              # Sealed executable object
│   ├── micronaut.ps1             # PowerShell orchestrator
│   ├── object.toml               # Object declaration
│   ├── semantics.xjson           # KUHUL-TSG schema
│   ├── brains/                   # Sealed n-gram data
│   ├── io/                       # Append-only I/O
│   ├── trace/                    # SCXQ2 evidence
│   └── proof/                    # Proof artifacts
├── src/                          # Python & JavaScript implementation
│   ├── fold_orchestrator.py      # Headless fold governor (15 fold types)
│   ├── verifier.py               # One-pass FEL verifier
│   ├── fel_svg_replay.py         # SVG replay generator
│   ├── fel_meta_fold.py          # META_FOLD attestation
│   ├── orchestrator_bot.py       # Bot orchestration system
│   ├── safetensors_cluster_orchestrator.py
│   ├── phi2_gguf_loader.js       # Phi-2 LLM loader with CM-1 gating
│   ├── inference_cluster.js      # 1000-node MoE inference grid (10x10x10)
│   ├── kuhul_3d_cluster.js       # 3D visualization with isometric projection
│   ├── cluster_node_state_machine.js  # DFA state machine for cluster nodes
│   ├── CLUSTER_NODE_DFA.md       # State machine specification
│   ├── test_inference_cluster.js # MoE routing validation tests
│   ├── test_cluster_node_dfa.js  # DFA test suite (10/10 passing)
│   └── ...
├── docs/
│   ├── fold_law.md               # Locked fold collapse rules
│   ├── swmsp_v1_s7_integration_mapping.md  # SWMSP v1.0.0 ↔ S7 shard/memory binding
│   ├── verifier_rules.md         # V0-V7 determinism rules
│   ├── control_micronaut_1.md    # CM-1 control alphabet spec
│   ├── scxq2_binary_packing_example.md
│   ├── ggl/                      # Plane-2 sealed compute artifacts
│   ├── fel-language-pack-v1/     # FEL v1 schema + ABI
│   └── golden_pack/              # Conformance test vectors
├── index.html                    # Entry point
└── BLUEPRINT.md                  # Extended design rationale
```

## Core Concepts

### Folds

15 typed domains, each with locked collapse rules:

```
⟁DATA_FOLD⟁      ⟁CODE_FOLD⟁      ⟁STORAGE_FOLD⟁
⟁NETWORK_FOLD⟁   ⟁UI_FOLD⟁        ⟁AUTH_FOLD⟁
⟁DB_FOLD⟁        ⟁COMPUTE_FOLD⟁   ⟁STATE_FOLD⟁
⟁EVENTS_FOLD⟁    ⟁TIME_FOLD⟁      ⟁SPACE_FOLD⟁
⟁META_FOLD⟁      ⟁PATTERN_FOLD⟁   ⟁CONTROL_FOLD⟁
```

See [`docs/fold_law.md`](docs/fold_law.md) for collapse rules.

### CSS Micronauts

Deterministic rule-sets that:
- Target a specific SVG scope
- Read state via CSS variables or attributes
- Emit no side effects outside their scope
- Can be toggled by the kernel

### SCXQ2 Lanes

Binary packing into five lanes (magic `SCX2`, CRC32-terminated):

| Lane | ID | Content | Folds |
|------|----|---------|-------|
| DICT | 1 | Symbol tables (names, IDs, opcodes) | DATA, AUTH, META, PATTERN |
| FIELD | 2 | Typed scalars (f32/q16/u32) | STORAGE, DB, STATE |
| LANE | 3 | Ordered events (canonical JSON) | UI, EVENTS, TIME |
| EDGE | 4 | Causality links / relations | CODE, NETWORK, SPACE, CONTROL |
| BATCH | 5 | Ephemeral compute jobs | COMPUTE |

### Cluster Node DFA (State Machine)

> **Scaling note:** the 1000-node cluster described here is a **logical runtime grid** used for deterministic routing and fold allocation. It is not 1000 CPU/GPU cores and not 1000 model replicas. For production serving, scale user concurrency with async APIs + batching first, then optionally add verified volunteer mesh compute for parallel side tasks.

Each cluster node is a Deterministic Finite Automaton (DFA) with:
- **6 states**: S0(IDLE) → S1(PARSE) → S2(TRANSFORM) → S3(EXECUTE) → S5(COMPLETE) or S4(ERROR)
- **Deterministic transitions**: δ(state, input) → (next_state, action, micronaut)
- **Pure compute semantics**: stack, tape, registers, program counter, accumulator
- **V2/V6 compliance**: control gate enforcement + deterministic hashing

The logical 1000-node cluster is fold-scoped with allocation:
- Compute nodes (500): Run MM-1 inference within ⟁COMPUTE_FOLD⟁
- Routing nodes (200): Route tokens via ngram matching
- Storage nodes (150): Seal results via SM-1
- Verification nodes (100): Attest proofs via VM-2
- Control nodes (50): Gate operations via CM-1

See [`src/CLUSTER_NODE_DFA.md`](src/CLUSTER_NODE_DFA.md) for full specification.
See [`docs/scaling_concurrency_vs_mesh.md`](docs/scaling_concurrency_vs_mesh.md) for the operational scaling model.

### LLM Model Format

MICRONAUT supports three model formats via `safetensors_cluster_orchestrator.py`. The `ModelFormat` enum selects the loader at runtime:

| Format | Enum Value | Use Case |
|--------|-----------|----------|
| **GGUF** | `ModelFormat.GGUF` | Primary format — quantized inference via transformers.js WASM |
| **SafeTensors** | `ModelFormat.SAFETENSORS` | HuggingFace native weights, zero-copy mmap loading |
| **PyTorch** | `ModelFormat.PYTORCH` | `.pt`/`.bin` checkpoint files, full-precision or quantized |

#### Primary Model: Phi-2 GGUF

The canonical model declared in `micronaut/models.toml`:

| Field | Value |
|-------|-------|
| Model | Phi-2 (Microsoft) |
| Parameters | 2.7B |
| Source | `TheBloke/phi-2-GGUF` |
| File | `models/phi-2/phi-2.Q2_K.gguf` |
| Tokenizer | `models/phi-2/tokenizer.json` |
| Config | `models/phi-2/config.json` |
| Quantization | Q2_K |
| Context length | 2048 tokens |
| Vocabulary | 51200 tokens |

#### Quantization Levels

The transformers.js WASM runtime supports four GGUF quantization types, ordered by quality vs. size trade-off:

| Quantization | Bits/weight | Notes |
|-------------|------------|-------|
| Q2_K | ~2.6 | Smallest — default for sealed deployment |
| Q4_K_M | ~4.8 | Balanced quality/size |
| Q5_K_M | ~5.7 | Near-lossless for most tasks |
| Q8_0 | ~8.5 | Near-full-precision |

#### Runtime and Inference

- **Runtime**: `transformers.js` WASM backend (browser and Node.js compatible)
- **Fold binding**: MM-1 micronaut → `⟁COMPUTE_FOLD⟁` → BATCH lane (lane 5)
- **Inference is fully deterministic**:

```
temperature = 0.0   # No sampling randomness
top_k       = 1     # Greedy decode
seed        = 42    # Fixed RNG seed
max_tokens  = 512   # Hard output limit
```

Because temperature is 0 and top_k is 1, every run with identical input produces byte-identical output. This is required for V6 replay determinism (same inputs → identical hashes).

#### Fold-Scoped Model Authority

The LLM is not a free agent — it is law-bound within the fold system:

```
Input text
  → NgramRouter (bigrams/trigrams → meta-intent-map.json)
  → MM-1 selected as active expert
  → CM-1 control gate must be open (V2 gate record required)
  → ComputeFold.process() wraps inference call
  → trace_hash recorded (input_hash + output_hash)
  → BATCH lane packs ephemeral result
  → State discarded (ComputeFold is ephemeral — no persistence)
```

Micronauts orchestrate and constrain the model; they never execute autonomously. KUHUL_π is the sole execution authority.

### CM-1 Control Alphabet

Pre-semantic control layer using Unicode C0 characters for phase signaling. See [`docs/control_micronaut_1.md`](docs/control_micronaut_1.md).

## Mixture of Experts (MoE) Architecture

The 9 micronauts operate as fold-scoped experts in a gated MoE system:

| Micronaut | Role | Fold | Tool Count | Ngram Triggers |
|-----------|------|------|-----------|----------------|
| CM-1 | Phase geometry | ⟁CONTROL_FOLD⟁ | 5 | control, gate, phase |
| PM-1 | Field selection | ⟁DATA_FOLD⟁ | 5 | perceive, parse, field |
| TM-1 | Collapse timing | ⟁TIME_FOLD⟁ | 5 | temporal, schedule, timer |
| HM-1 | Host abstraction | ⟁STATE_FOLD⟁ | 5 | host, state, abstract |
| SM-1 | Inert persistence | ⟁STORAGE_FOLD⟁ | 5 | storage, seal, persist |
| MM-1 | Token signal generation | ⟁COMPUTE_FOLD⟁ | 5 | model, token, inference |
| XM-1 | Narrative expansion | ⟁PATTERN_FOLD⟁ | 5 | pattern, expand, narrative |
| VM-1 | Rendering projection | ⟁UI_FOLD⟁ | 5 | visual, render, project |
| VM-2 | Proof generation | ⟁META_FOLD⟁ | 5 | verify, proof, attest |

**MoE Routing**: Input text is scored against bigrams (weight 2) and trigrams (weight 3) from `meta-intent-map.json`. Highest-scoring micronaut is selected as the active expert.

**9-Stage Pipeline**:
1. PM-1 (Perceive): Field selection from input
2. CM-1 (Gate): Control gate opening with V2 authorization
3. TM-1 (Schedule): Temporal alignment and fold-scoped timing
4. HM-1 (Normalize): Host state normalization
5. MM-1 (Infer): Token generation via Phi-2 GGUF model
6. XM-1 (Expand): Pattern-based narrative expansion
7. SM-1 (Seal): Result sealing and persistence
8. VM-2 (Verify): Proof generation and attestation
9. VM-1 (Render): SVG/CSS projection for output

## Key Specifications

| Document | Description |
|----------|-------------|
| [`docs/fold_law.md`](docs/fold_law.md) | Fold collapse rules (locked) |
| [`docs/micronaut_fold_law_unified.md`](docs/micronaut_fold_law_unified.md) | Unified fold algebra + frozen law artifacts |
| [`docs/verifier_rules.md`](docs/verifier_rules.md) | V0-V7 determinism invariants |
| [`docs/control_micronaut_1.md`](docs/control_micronaut_1.md) | CM-1 control alphabet |
| [`docs/fold_lattice_proofs.md`](docs/fold_lattice_proofs.md) | Capability lattice proofs |
| [`docs/scxq2_binary_packing_example.md`](docs/scxq2_binary_packing_example.md) | SCXQ2 frame format |
| [`docs/atomic_dom_binary_ingest.md`](docs/atomic_dom_binary_ingest.md) | ATOMIC-DOM binary ingest pipeline |
| [`src/CLUSTER_NODE_DFA.md`](src/CLUSTER_NODE_DFA.md) | Cluster node DFA specification |
| [`CLAUDE.md`](CLAUDE.md) | LLM integration model and MoE architecture |

## Verifier Rules

| Rule | Invariant |
|------|-----------|
| V0 | Deterministic (no randomness) |
| V2 | Control gate monopoly (all mutations require gate record) |
| V3 | Replayable from identical input |
| V4 | No floating-point in proofs |
| V7 | Phase boundaries align with SCXQ2 frames |

## Usage

### FEL Verification

```bash
# Verify a FEL event stream
python src/verifier.py events.fel.jsonl

# With SVG replay output
python src/verifier.py events.fel.jsonl --out replay_out --write-svg

# With binary packing
python src/verifier.py events.fel.jsonl --write-bin

# Validate frozen fold law input (default input: system.json)
python src/fold_verifier.py system.json
```

### MoE Inference Cluster

```bash
# Run MoE routing tests (100% accuracy)
node src/test_inference_cluster.js

# Run cluster node DFA tests (10/10 passing)
node src/test_cluster_node_dfa.js

# Example: Execute on compute node
const orchestrator = new ClusterStateMachineOrchestrator(1000);
const result = orchestrator.executeOnNodeType('compute', ['parse', 'transform', 'execute', 'complete']);
console.log(result.finalState);      // S5 (complete)
console.log(result.accumulator);     // Final computed value
console.log(result.deterministic);   // true
```

### Phi-2 Token Generation

```bash
// Initialize loader with CM-1 control gate
const loader = new Phi2GGUFLoader({
  modelPath: 'models/phi-2-gguf-q2_k.gguf',
  boundMicronaut: 'MM-1',
  boundFold: '⟁COMPUTE_FOLD⟁',
  defaultParams: { temperature: 0.0, top_k: 1, seed: 42 }
});

// Open control gate for MM-1 operations
loader.openControlGate({
  decide_hash: '...',
  policy_hash: '...'
});

// Generate token (deterministic)
const result = await loader.emitToken('What is Micronaut?');
console.log(result.token);           // Single generated token
console.log(result.hash);            // SHA-256 of output
```

## Properties

- **Deterministic**: CSS variable math, static SVG structure, append-only frames
- **Verifiable**: Proofs replay without executing models
- **Replaceable**: Swap sealed executors; proofs still validate
- **Compressible**: SCXQ2 crushes state into symbolic lanes

## Related

- [MATRIX](https://github.com/cannaseedus-bot/MATRIX) — Universal language server bridge

## Development Phases

### ✅ Completed
- [x] **Phase 1**: Sealed data architecture (object.toml, semantics.xjson, brains/)
- [x] **Phase 2**: 15-fold collapse law specification and FEL verifier (V0-V7 rules)
- [x] **Phase 3**: Micronauts registry and role definitions (9 micronauts, fold scoping)
- [x] **Phase 4**: Job-specific ngram tools for micronauts (bigrams, trigrams, intent routing)
- [x] **Phase 5**: SVG/CSS deterministic state machine implementation
  - ClusterNodeDFA: 6-state DFA with stack, tape, registers
  - V2 control gate enforcement
  - V6 deterministic hashing
  - 10/10 test coverage (100%)
- [x] **Phase 5b**: LLM integration layer (Phi-2 GGUF + MoE cluster)
  - Phi-2 GGUF loader with CM-1 gating
  - 1000-node 3D cluster (10x10x10 grid)
  - 9-micronaut MoE expert routing via ngrams
  - 3D visualization with isometric projection
  - Deterministic inference (temperature=0, seed=42)

- [x] **Phase 6**: V2/V3 verifier gap-fill (`--strict-v2`, `--strict-v3` flags added to `verifier.py`)
- [x] **Phase 7**: SCXQ2 full 5-lane binary packer (`src/scxq2_packer.py`, fold→lane routing table)
- [x] **Phase 8**: CM-1 control alphabet pre-semantic layer (`src/cm1_parser.py`, 5 invariants CM1-S1–S5)
- [x] **Phase 9**: Multi-worker bot orchestration and fold routing (`src/orchestrator_bot.py` fold dispatch + `src/ngram_router.py`)
- [x] **Phase 10**: GGL sealed compute integration (`src/ggl_orchestrator.py`, `src/scxq2_ggl_packer.py`, 10/10 bootstrap vectors)
- [x] **Phase 11**: VM-1 rendering projections — CSS, DOM, and terminal renderers added to UIFold
- [x] **Phase 12**: Golden pack conformance maintained throughout all phases (SVG sha256 `905fa675…`, binary sha256 `544e2899…` unchanged)

## License

See repository for license information.
