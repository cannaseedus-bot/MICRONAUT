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
⟁EVENTS_FOLD⟁    ⟁TIME_FOLD⟁      ⟁ERROR_FOLD⟁
⟁CONFIG_FOLD⟁    ⟁PROOF_FOLD⟁     ⟁CONTROL_FOLD⟁
```

See [`docs/fold_law.md`](docs/fold_law.md) for collapse rules.

### CSS Micronauts

Deterministic rule-sets that:
- Target a specific SVG scope
- Read state via CSS variables or attributes
- Emit no side effects outside their scope
- Can be toggled by the kernel

### SCXQ2 Lanes

Binary packing into four lanes:

| Lane | Content |
|------|---------|
| DICT | Symbol table (names, IDs, opcodes) |
| FIELD | Typed scalars (f32/q16/u32) |
| LANE | Payload blocks (canonical JSON, b64) |
| EDGE | Causality links (prev_hash, call_id) |

### Cluster Node DFA (State Machine)

Each cluster node is a Deterministic Finite Automaton (DFA) with:
- **6 states**: S0(IDLE) → S1(PARSE) → S2(TRANSFORM) → S3(EXECUTE) → S5(COMPLETE) or S4(ERROR)
- **Deterministic transitions**: δ(state, input) → (next_state, action, micronaut)
- **Pure compute semantics**: stack, tape, registers, program counter, accumulator
- **V2/V6 compliance**: control gate enforcement + deterministic hashing

The 1000-node cluster is fold-scoped with allocation:
- Compute nodes (500): Run MM-1 inference within ⟁COMPUTE_FOLD⟁
- Routing nodes (200): Route tokens via ngram matching
- Storage nodes (150): Seal results via SM-1
- Verification nodes (100): Attest proofs via VM-2
- Control nodes (50): Gate operations via CM-1

See [`src/CLUSTER_NODE_DFA.md`](src/CLUSTER_NODE_DFA.md) for full specification.

### Phi-2 GGUF LLM Integration

A lightweight deterministic LLM control plane:
- **Model**: Phi-2 2.7B (Q2_K quantized GGUF)
- **Binding**: MM-1 micronaut within ⟁COMPUTE_FOLD⟁
- **Inference**: Deterministic (temperature=0, top_k=1, seed=42)
- **Tools**: 5 MM-1 tools (emit_token, stream_tokens, voice_model, score_logits, sample_distribution)
- **Routing**: 9 micronauts as MoE experts via ngram-based gating

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

### 📋 Planned
- [ ] **Phase 6**: PowerShell orchestrator (micronaut.ps1) full integration
- [ ] **Phase 7**: SCXQ2 binary lane packing and replay verification
- [ ] **Phase 8**: CM-1 control alphabet pre-semantic layer
- [ ] **Phase 9**: Multi-worker bot orchestration and fold routing
- [ ] **Phase 10**: Token streaming and adaptive batching for MM-1
- [ ] **Phase 11**: Proof generation and attestation chains (META_FOLD)
- [ ] **Phase 12**: Production verifier and golden pack conformance

## License

See repository for license information.
