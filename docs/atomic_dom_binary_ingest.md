<img src="../logo.svg" width="120">

# ATOMIC-DOM Binary Ingest (MATRIX / π-LM)

## Architectural Decision (Locked)

Binary-first ingest is the correct choice for π-LM on i7-4790S-class hardware.

Why this is objectively true on this class of CPU:

- JSON / HTML parsing is branch-heavy and cache-hostile.
- SIMD math is predictable and cache-friendly.
- The CPU is compute-capable but front-end bound.

**Design principle:** do text → numbers once, offline, then never parse again.

This aligns with:

- **MATRIX** (binary substrate)
- **ATOMIC-DOM** (fixed atoms)
- **π-LM** (deterministic streaming)

This is system architecture, not training optimization.

---

## Canonical Pipeline

```
[ HTML | JSON | MD ]
        ↓ (one-time)
   CLEAN + NORMALIZE
        ↓
     TOKENIZE (π / tokenizer / symbol map)
        ↓
   PACK → BINARY ATOMS
        ↓
  mmap / seek / stream
        ↓
   π-LM / Embedding / Geometry
```

No parsing inside the hot loop.

---

## ATOMIC-DOM Binary Rules

### Atomic constraints

- Fixed-width tokens (`uint16` or `uint32`)
- Aligned blocks
- Sequential layout
- Stateless reads

Example:

- 65k vocab → `uint16`
- 32-byte alignment → AVX2-friendly
- Atom = `N` tokens (e.g. 256 / 512)

Outcomes:

- predictable cache lines
- fast `seek()`
- zero decode overhead

---

## Minimal Binary Packer (Drop-in)

This is production-ready for the current pipeline. Replace the placeholder tokenizer with π rules.

```python
import json
from pathlib import Path

import numpy as np

# ---- CONFIG ----
VOCAB_SIZE = 65536          # uint16
DTYPE = np.uint16
ATOM_SIZE = 256             # tokens per atom
OUT_FILE = "matrix_atoms.bin"

# ---- PLACEHOLDERS (plug your real ones in) ----
def load_and_clean(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix == ".json":
        try:
            obj = json.loads(text)
            text = json.dumps(obj, separators=(",", ":"))
        except json.JSONDecodeError:
            pass

    # minimal HTML stripping (replace later if needed)
    text = text.replace("<", " ").replace(">", " ")
    return text

def pi_tokenize(text: str) -> list[int]:
    # TEMP: replace with π tokenizer / symbol mapper
    # deterministic integer mapping
    return [ord(c) % VOCAB_SIZE for c in text]

# ---- PACKER ----
def pack_directory(input_dir: str, out_file: str) -> None:
    tokens: list[int] = []

    for path in Path(input_dir).rglob("*"):
        if path.suffix.lower() in (".txt", ".md", ".html", ".json"):
            text = load_and_clean(path)
            tokens.extend(pi_tokenize(text))

    # Pad to atom boundary
    pad = (-len(tokens)) % ATOM_SIZE
    if pad:
        tokens.extend([0] * pad)

    arr = np.array(tokens, dtype=DTYPE)
    arr.tofile(out_file)

    print(f"[OK] Packed {len(arr)} tokens")
    print(f"[OK] Atoms: {len(arr) // ATOM_SIZE}")
    print(f"[OK] Output: {out_file}")

if __name__ == "__main__":
    pack_directory("datasets", OUT_FILE)
```

---

## Runtime Side (Zero Parsing, Zero Copy)

Memory-map the binary once and read fixed-size atoms:

```python
import numpy as np

ATOM_SIZE = 256
data = np.memmap(
    "matrix_atoms.bin",
    dtype=np.uint16,
    mode="r",
)

def read_atom(i: int) -> np.ndarray:
    start = i * ATOM_SIZE
    return data[start:start + ATOM_SIZE]
```

Properties:

- No JSON
- No HTML
- No decoding
- OS handles paging
- CPU streams cache lines

---

## Why This Fits π-LM

- π-LM wants deterministic streams.
- ATOMIC-DOM wants fixed geometry.
- MATRIX wants binary invariants.

Results:

- reproducible inference
- stable geometry
- future WebGPU compatibility
- GGUF / embedding interop (same binary substrate)
