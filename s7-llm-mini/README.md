# S7-LLM-MINI-1M

Deterministic Fold-Compliant INT8 Transformer — SCXQ2 Sealed Artifact (`.s7l`)

## What It Is

A fully runnable minimal reference implementation of a `.s7l`-sealed LLM inference engine:

- **Pure Rust CPU** deterministic inference
- **INT8 quantized** weights (no floating-point in proofs)
- **`.s7l` sealed artifact** — lane-hash verified, Merkle rooted
- **Greedy decode only** — no randomness, no sampling
- **Zero ML framework dependencies** — only `sha2` and `serde_json`
- **Fold-law compliant** — bound to `⟁COMPUTE_FOLD⟁` via MM-1 micronaut

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run
```

## Layout

```
s7-llm-mini/
├── Cargo.toml
├── build.rs
├── model/
│   ├── mini.s7l          # Sealed model artifact (S7LM magic, lane-hash verified)
│   └── vocab.json        # Deterministic tokenizer vocabulary
└── src/
    ├── main.rs
    ├── s7l/              # .s7l file format (header, lanes, Merkle)
    ├── tensor/           # INT8 tensor type
    ├── model/            # Embedding, Linear, Attention, FFN, Transformer
    ├── tokenizer/        # BPE tokenizer (vocab.json-backed)
    └── inference/        # Greedy decode
```

## `.s7l` File Format

```
Offset  Size  Field
------  ----  -----
0       4     Magic: "S7LM"
4       2     Version (big-endian u16)
6       1     Class (0x01 = LLM)
7       1     Flags
8       32    Root Merkle hash (SHA-256 over all lane hashes)
40+     ...   Lanes (repeating):
                1 byte  lane ID
                4 bytes payload length (big-endian u32)
                N bytes payload
                32 bytes SHA-256 hash of payload (verified on load)
```

Lane IDs map to SCXQ2 lanes:

| ID | Lane  | Content              |
|----|-------|----------------------|
| 1  | DICT  | Vocabulary / symbols |
| 2  | FIELD | INT8 weight tensors  |
| 3  | LANE  | Generation stream    |
| 4  | EDGE  | CM-1 topology        |
| 5  | BATCH | Ephemeral compute    |

## Guarantees

- Deterministic inference (greedy argmax, no RNG)
- Lane-hash verified on parse (SHA-256, panics on mismatch)
- Merkle root seals the full artifact
- CM-1 governed (bound fold: `⟁COMPUTE_FOLD⟁`)
- V6 compliant: same input → byte-identical output
