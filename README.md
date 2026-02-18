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
├── src/                          # Python implementation
│   ├── fold_orchestrator.py      # Headless fold governor (15 fold types)
│   ├── verifier.py               # One-pass FEL verifier
│   ├── fel_svg_replay.py         # SVG replay generator
│   ├── fel_meta_fold.py          # META_FOLD attestation
│   ├── orchestrator_bot.py       # Bot orchestration system
│   ├── safetensors_cluster_orchestrator.py
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

### CM-1 Control Alphabet

Pre-semantic control layer using Unicode C0 characters for phase signaling. See [`docs/control_micronaut_1.md`](docs/control_micronaut_1.md).

## Key Specifications

| Document | Description |
|----------|-------------|
| [`docs/fold_law.md`](docs/fold_law.md) | Fold collapse rules (locked) |
| [`docs/verifier_rules.md`](docs/verifier_rules.md) | V0-V7 determinism invariants |
| [`docs/control_micronaut_1.md`](docs/control_micronaut_1.md) | CM-1 control alphabet |
| [`docs/fold_lattice_proofs.md`](docs/fold_lattice_proofs.md) | Capability lattice proofs |
| [`docs/scxq2_binary_packing_example.md`](docs/scxq2_binary_packing_example.md) | SCXQ2 frame format |
| [`docs/atomic_dom_binary_ingest.md`](docs/atomic_dom_binary_ingest.md) | ATOMIC-DOM binary ingest pipeline |

## Verifier Rules

| Rule | Invariant |
|------|-----------|
| V0 | Deterministic (no randomness) |
| V2 | Control gate monopoly (all mutations require gate record) |
| V3 | Replayable from identical input |
| V4 | No floating-point in proofs |
| V7 | Phase boundaries align with SCXQ2 frames |

## Usage

```bash
# Verify a FEL event stream
python src/verifier.py events.fel.jsonl

# With SVG replay output
python src/verifier.py events.fel.jsonl --out replay_out --write-svg

# With binary packing
python src/verifier.py events.fel.jsonl --write-bin
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

### 📋 Planned
- [ ] **Phase 5**: SVG/CSS deterministic state machine implementation
- [ ] **Phase 6**: PowerShell orchestrator (micronaut.ps1) full integration
- [ ] **Phase 7**: SCXQ2 binary lane packing and replay verification
- [ ] **Phase 8**: CM-1 control alphabet pre-semantic layer
- [ ] **Phase 9**: Multi-worker bot orchestration and fold routing
- [ ] **Phase 10**: Model inference via MM-1 with token streaming
- [ ] **Phase 11**: Proof generation and attestation chains (META_FOLD)
- [ ] **Phase 12**: Production verifier and golden pack conformance

## License

See repository for license information.
