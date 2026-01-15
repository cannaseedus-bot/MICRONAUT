# Golden pack (fold → lane → deterministic container)

This directory contains the pinned golden vector pack plus the single-pass verifier.

## Contents

* `events.jsonl` — canonical input events for the golden pack.
* `verify_golden_pack.py` — one-pass verifier that asserts invariants and regenerates outputs.
* `svg_replay.py` — standalone SVG replay generator for ad-hoc inspection.

## Expected hashes

* `replay_0001.svg` (SHA-256):
  `905fa675040e66d9f9695b71da669c961e13822e476328fb20b0b6961fb3aeba`
* `scx2.bin` (SHA-256):
  `544e2899b93c9e0e26f3092d94aafc5ee194a47a066b8db91172326c8846613d`

## Run the verifier

```bash
python verify_golden_pack.py events.jsonl --write
```

The verifier writes:

* `replay_0001.svg` (derived projection)
* `scx2.bin` (packed lanes)

Use `svg_replay.py` for multi-frame replay output:

```bash
python svg_replay.py events.jsonl --out replay
```
