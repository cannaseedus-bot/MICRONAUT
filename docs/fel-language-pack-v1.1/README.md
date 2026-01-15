# FEL — Fold Event Language v1.1

Canonical ID: asx://language/fel/v1.1
Schema ID: kuhul://schema/fel/v1.1
Schema authority: xjson://schema/core/v1 (offline, internal)

FEL v1.1 is a backward-compatible extension of FEL v1:

- canonical serialization (one true byte stream per meaning)
- legality rules (invalid programs rejected)
- deterministic replay (same input → same state)
- delta/patch semantics for state evolution
- storage seals with lane binding
- attestation chains for proof lineage

## Commands

### 1) Verify the full pack (invariants + expected hashes)

```bash
python tools/verify_fel_pack.py vectors/v1.1/events.jsonl --write-out
```

### 2) Compile FEL → SCX2 binary container

```bash
python tools/felc.py vectors/v1.1/events.jsonl --out scx2.bin
```

### 3) Project FEL → SVG replay frames

```bash
python tools/felp.py vectors/v1.1/events.jsonl --out replay_out
```

## Golden vector expectations (SHA-256)

See vectors/v1.1/expected.json.
