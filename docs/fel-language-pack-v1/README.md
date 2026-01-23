# FEL — Fold Event Language v1

Canonical ID: asx://language/fel/v1
Schema ID: kuhul://schema/fel/v1
Schema authority: xjson://schema/core/v1 (offline, internal)

FEL is a deterministic, canonical JSONL event language governed by Fold Law:

- canonical serialization (one true byte stream per meaning)
- legality rules (invalid programs rejected)
- deterministic replay (same input → same state)
- compilable to lane-packed SCX2 container
- projectable to SVG replay frames (read-only projection)

## Commands

### 1) Verify the full pack (invariants + expected hashes)

```bash
python tools/verify_fel_pack.py vectors/v1/events.jsonl --write-out
```

### 2) Compile FEL → SCX2 binary container

```bash
python tools/felc.py vectors/v1/events.jsonl --out scx2.bin
```

### 3) Project FEL → SVG replay frames

```bash
python tools/felp.py vectors/v1/events.jsonl --out replay_out
```

## Golden vector expectations (SHA-256)

See vectors/v1/expected.json.
