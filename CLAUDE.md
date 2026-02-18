# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MICRONAUT is a deterministic, proof-driven SCO/1 object server. It treats web primitives (SVG, CSS, JSON) as computation infrastructure rather than presentation. The system enforces a sealed-data, file-based architecture where correctness is decided by a verifier, not runtime.

## Key Commands

### Verify a FEL event stream
```bash
python src/verifier.py events.fel.jsonl
python src/verifier.py events.fel.jsonl --out replay_out --write-svg --write-bin
```

### Run golden pack conformance test
```bash
python docs/golden_pack/verify_golden_pack.py docs/golden_pack/events.jsonl --write
```

### FEL v1.1 language pack tools
```bash
python docs/fel-language-pack-v1.1/tools/verify_fel_pack.py vectors/v1.1/events.jsonl --write-out
python docs/fel-language-pack-v1.1/tools/felc.py vectors/v1.1/events.jsonl --out scx2.bin   # Compile FEL → SCX2 binary
python docs/fel-language-pack-v1.1/tools/felp.py vectors/v1.1/events.jsonl --out replay_out  # Project FEL → SVG frames
```

### SVG replay generation
```bash
python docs/golden_pack/svg_replay.py docs/golden_pack/events.jsonl --out replay
```

## Architecture

### Three-Plane Model

- **Plane 0/1 (Control & Law)**: `micronaut/object.toml` declares the object server; `micronaut/semantics.xjson` defines the KUHUL-TSG schema; `micronaut/folds.toml` is the unified fold/micronaut/routing metadata. These are the law layer — no execution happens here.
- **Plane 2 (Sealed Compute)**: `micronaut/brains/` contains read-only sealed data (ngram profiles, bigrams, trigrams, intent maps). `micronaut/micronaut.s7` is the sealed executable object.
- **Plane 3 (Proof & Replay)**: `micronaut/io/` has append-only I/O (chat.txt, stream.txt). `micronaut/trace/` and `micronaut/proof/` hold evidence and proof artifacts.

### FEL-TOML Metadata (`micronaut/folds.toml`)

The authoritative single source of truth for the fold system. Consolidates:
- All 15 fold declarations with collapse rules, SCXQ2 lane mappings, and CM-1 phase bindings
- All 9 micronaut registrations with their 5 job-specific tools and ngram triggers
- Intent routing table for ngram-based tool selection (fallback: XM-1)
- CM-1 control phase mappings (U+0000–U+0004 Unicode C0 characters)
- Verifier rules V0–V7 declarations

### SMCA Authority Automaton (`smca/`)

A DFA-based authority model defining the legal state transitions for operations. States flow: MATRIX_PROPOSAL → CM1_GATE → SCXQ7_LAW → SCXQ2_SEMANTIC → SCO_EXECUTION → IDB_COMMIT. Any illegal transition lands in S⊥ (rejected). Contains SVG diagrams for cluster roles, collapse geometry, and authority gradient.

### 15-Fold System

All operations are scoped to exactly one of 15 typed domains ("folds"). Each fold has one legal collapse rule and maps to one SCXQ2 lane. The authoritative spec is `docs/fold_law.md`. Key constraint: folds cannot re-expand after collapse without replay.

The three sovereignty folds are: CONTROL_FOLD (permits execution), STORAGE_FOLD (seals persistence), UI_FOLD (projects output). Everything else routes through them.

### 9 Micronauts

Defined in `micronaut/micronaut.registry.xjson` (structural definitions) and `micronaut/brains/micronaut-profiles.json` (runtime profiles with ngram tools). Each micronaut is role-scoped and law-bound:

| ID | Role | Fold |
|----|------|------|
| CM-1 | phase_geometry | CONTROL_FOLD |
| PM-1 | field_selection | DATA_FOLD |
| TM-1 | collapse_timing | TIME_FOLD |
| HM-1 | host_abstraction | STATE_FOLD |
| SM-1 | inert_persistence | STORAGE_FOLD |
| MM-1 | token_signal_generator | COMPUTE_FOLD |
| XM-1 | narrative_expansion | PATTERN_FOLD |
| VM-1 | rendering_projection | UI_FOLD |
| VM-2 | proof_generation | META_FOLD |

Each micronaut has tools with ngram triggers (bigrams/trigrams) that activate based on input text matching. Tool routing is defined in `micronaut/brains/meta-intent-map.json`.

### Verifier Rules (V0-V7)

The verifier (`src/verifier.py`) enforces determinism in a single pass. Spec is at `docs/verifier_rules.md`. Critical rules:
- **V0**: Canonical JSON ordering (keys sorted lexicographically)
- **V2**: All mutations require explicit control gate records
- **V5**: Fold-to-lane mapping is fixed; mismatches are rejected
- **V6**: Same inputs must produce identical collapse/snapshot/binary hashes

### SCXQ2 Lanes

Binary packing into 5 lanes: DICT (symbols), FIELD (typed scalars), LANE (payload blocks), EDGE (causality links), BATCH (ephemeral compute). Fold→lane mapping is deterministic and fixed.

### Key Source Files

- `src/verifier.py` — FEL v1.1 one-pass verifier (the correctness authority)
- `src/fold_orchestrator.py` — Headless fold governor enforcing collapse laws
- `src/fel_svg_replay.py` — SVG replay frame generator
- `src/fel_meta_fold.py` — META_FOLD attestation chain executor
- `src/orchestrator_bot.py` — Bot orchestration with worker spawning
- `src/safetensors_cluster_orchestrator.py` — Model loader for SafeTensors/PyTorch/GGUF formats
- `src/kuhul_micronaut_factory.js` — Agent spawning and capability matrix (JS)
- `src/micronaut_handlers.js` — N-gram inference handlers (JS)

### LLM Integration Model

The system is designed as a **control plane around an LLM**, not a standalone model:
- **MM-1 (ModelMicronaut)** handles token signal generation within `⟁COMPUTE_FOLD⟁`
- `safetensors_cluster_orchestrator.py` supports SafeTensors, PyTorch, and GGUF model formats
- The micronaut registry references `phi-2-sft-lora`, `distilgpt2`, `gguf_models` as candidate model types
- Micronauts route, constrain, and verify LLM output — they never execute autonomously
- The ngram system (bigrams/trigrams/intent-map) provides lightweight local inference for tool routing independent of the LLM

## Important Conventions

- **Sealed data is immutable**: Files under `micronaut/brains/` are read-only sealed data. The `mutation: "forbidden"` field in JSON schemas is a law constraint, not a suggestion.
- **Append-only I/O**: `micronaut/io/chat.txt` and `micronaut/io/stream.txt` are append-only; never truncate or overwrite.
- **xjson schema files**: Files with `.xjson` extension follow the project's extended JSON schema convention with `@schema`, `@status`, `@authority`, and `@mutation` fields.
- **Fold delimiters**: Folds are referenced using the `⟁FOLD_NAME⟁` Unicode delimiter syntax throughout the codebase.
- **No runtime authority**: Micronauts orchestrate and project but never execute autonomously. KUHUL_π is the sole execution authority.
- **Proof over execution**: The verifier decides correctness. If the verifier can't reproduce identical hashes from the same inputs, the system is broken.
- **folds.toml is authoritative**: `micronaut/folds.toml` is the single source of truth for fold declarations, micronaut tools, ngram routing, and CM-1 phases. Keep it in sync with the JSON brain files.
