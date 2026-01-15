# FEL Verifier v1.1 — Minimal Law README (Verifier-Grade)

This repository contains a **one-pass verifier** for **FEL v1.1** event streams and their derived artifacts:
- Deterministic **STATE roots** per tick
- Deterministic **SVG replay frames** per tick (projection-only)
- Optional deterministic **META_FOLD attestation chain**
- Deterministic **SCX2 binary commit** (`scx2.bin`)

Core thesis: **FEL is a language because it has**:
1) a grammar,
2) a deterministic small-step state transition relation,
3) canonical hashing rules,
4) replay/projection determinism,
5) verifier-enforced invariants.

---

## Files

- `verifier.py`
  One-pass verifier. Reads `events.fel.jsonl`, replays state, projects SVG, optionally generates/validates META attestations, optionally emits `scx2.bin`.

Outputs (depending on flags):
- `verify_manifest.json` (always)
- `replay_0001.svg ... replay_N.svg` (`--write-svg`)
- `scx2.bin` (`--write-bin`)

---

## FEL v1.1 Objects (minimum)

Each JSONL line is one FEL event:

Required fields:
- `v`: `"fel.v1"` or `"fel.v1.1"`
- `tick`: integer `>= 0`
- `type`: `"set" | "delta" | "emit" | "seal" | "attest" | ..."`
- `fold`: e.g. `"⟁STATE_FOLD⟁"`

STATE mutation forms:
- `type="set"`, `fold="⟁STATE_FOLD⟁"`, fields: `key`, `value`
- `type="delta"`, `fold="⟁STATE_FOLD⟁"`, fields: `patch{op,path,value?}`

META attest (optional in FEL input):
- `type="attest"`, `fold="⟁META_FOLD⟁"`, fields:
  - `policy_hash`, `abi_hash`, `meta_hash`, `prev?`, `att_hash?`

---

## Canonical Hashing Law (the root of determinism)

All verifier hashes use:
- `SHA256(canon_json(obj))`

Where `canon_json` is:
1) **Remove observational keys** (currently: `timestamp`) recursively
2) **Deterministic ordering**
   - top-level keys are ordered by a fixed list (then remaining keys sorted)
   - nested dict keys are sorted
3) JSON encoding is stable:
   - UTF-8
   - separators `(",", ":")`
   - no whitespace

This ensures:
- identical logical objects ⇒ identical bytes ⇒ identical hashes
- replay artifacts can be verified across machines/languages

**Implementation:** `canon_json()` + `sha256_hex()` in `verifier.py`.

---

## Formal Invariants (Math → Code Mapping)

Notation:
- Event stream: `E = [e0, e1, ...]`
- Tick partition: `E_t = { e ∈ E | e.tick = t }`
- State at tick t: `S_t` (a JSON object)
- Transition relation: `S_{t+} = δ(S_t, E_t)` (apply all mutations in tick t)
- State root: `R_t = H(S_{t+})`
- SVG projection: `V_t = Π(t, E_t, S_{t+})`
- SVG hash: `VH_t = H(V_t)`
- Attestation chain: `A_t` with head hash `AH_t`

Where:
- `H(x) = SHA256(canon_json(x))` for JSON objects
- `H(svg) = SHA256(utf8(svg_string))`
- `Π` is deterministic (no randomness, no time, no external IO)

### I1 — Tick Monotonicity
**Law:** `tick(e_i) ≤ tick(e_{i+1})`

**Why:** Deterministic folding requires a total time order.

**Code:** `validate_fel_line()` checks `tick < last_tick` fails.

---

### I2 — Fold Legality (No forbidden targets)
**Law:** Events must not target CONTROL/UI folds:
- `fold(e) ∉ { ⟁CONTROL_FOLD⟁, ⟁UI_FOLD⟁ }`

**Why:** CONTROL is governance; UI is projection. Neither is a mutation target.

**Code:** `FORBIDDEN_TARGET_FOLDS` + check in `validate_fel_line()`.

---

### I3 — STATE-Only Mutation
**Law:** Only `⟁STATE_FOLD⟁` may mutate `S`:
- if `type ∈ {set, delta}` then `fold = ⟁STATE_FOLD⟁`

**Why:** Prevents hidden state changes through “UI” or “CODE” events.

**Code:** checks in `validate_fel_line()`, and transitions applied only for `set/delta`.

---

### I4 — Deterministic Small-Step Semantics (STATE)
Define transition function for a tick:
- `δ(S, E_t) = fold(apply, S, M_t)`
- where `M_t = [m ∈ E_t | m.type ∈ {set, delta}]` in original order

Set:
- `apply_set(S, key, value)` does `S[key] := value`

Delta patch:
- `apply_delta(S, patch)` supports:
  - `op ∈ {add, replace}`: set path to value
  - `op = remove`: delete path key if present
- with path semantics using `/a/b/c` JSON object descent

**Why:** This is the “execution” of FEL: **state changes only**.

**Code:** `apply_set()`, `apply_delta()`.

---

### I5 — State Root Determinism
**Law:** `R_t = H(S_t)` after applying all tick mutations.

**Why:** Gives a stable commit root (verifier-grade).

**Code:** `state_root_sha256()`.

---

### I6 — Projection Determinism (UI as a function)
**Law:** `V_t = Π(t, E_t, S_t)` and Π is pure:
- no clocks
- no randomness
- no external reads

**Guarantee:** Same `E_t` + same `S_t` ⇒ same SVG bytes ⇒ same `VH_t`.

**Code:** `render_svg()` uses only `(tick, events, state)` and fixed layout.

---

### I7 — META_FOLD Attestation Chain (optional)
If META mode enabled (or attest lines exist), chain law:

**Pinning (no drift):**
- `policy_hash` constant across stream
- `abi_hash` constant across stream

**Chain:**
- For genesis tick: `prev ∈ {null, 64x"0"}`
- For subsequent: `prev = AH_{t-1}`

**Att hash:**
- `AH_t = H(attest_line(t, policy_hash, abi_hash, meta_hash, prev))`

**Meta hash (recommended):**
- `meta_hash = H({domain, stream_id, state_root_sha256, lane_roots, extra})`

**Why:** Produces a replayable proof chain without trusting runtimes.

**Code:**
- `verify_attestations_present()` checks pinning + chain + att hash
- generation mode: `--meta` computes `meta_hash` and `attestation_sha256`

---

### I8 — Deterministic SCX2 Binary Commit
Given per-tick records, pack bytes exactly (format is fixed), then:
- `BIN = pack_scx2(records)`
- `BIN_HASH = SHA256(BIN)`

**Why:** A binary “commit artifact” suitable for storage/distribution.

**Code:** `pack_scx2()` + `sha256_hex(scx2_bytes)`.

---

## What Makes the Verifier Sound

“Sound” here means: **if the verifier passes, the derived artifacts are uniquely determined by the FEL stream under the language law.**

The verifier is sound because:

1) **All entropy sources are excluded**
   - timestamps stripped from canonical hashing
   - projection function uses no external data

2) **State transitions are explicit and total**
   - only two mutation primitives (`set`, `delta`)
   - patch semantics are validated (op/path rules)

3) **Forbidden authority channels are blocked**
   - CONTROL and UI cannot be targeted
   - STATE is the only mutation locus

4) **Canonical hashing commits semantics, not formatting**
   - different JSON key orderings normalize to same bytes
   - same meaning ⇒ same hash

5) **Proof chain is pinned**
   - ABI + policy hashes cannot drift
   - `prev` links enforce total order of commitments

---

## Replay + Proof Guarantees

Given:
- the same `events.fel.jsonl`
- the same policy/abi pins (if META mode)

The verifier guarantees:

### G1 — Deterministic State Roots
For every tick, it produces `state_root_sha256` that must match across machines.

### G2 — Deterministic SVG Frames
For every tick, it produces `replay_####.svg` whose SHA256 is stable.

### G3 — Deterministic Commit Binary
If enabled, it produces `scx2.bin` whose SHA256 is stable.

### G4 — Deterministic Attestation Head
If META mode enabled, it produces an attestation chain head hash `meta_head_sha256` that is stable.

### G5 — One-Pass Verifiability
All invariants are asserted in a single pass:
- parse → validate → replay → project → attest → pack → hash → assert

---

## CLI Usage

Basic verify (no files written):
```bash
python verifier.py events.fel.jsonl
```

Write SVG + SCX2 binary:

```bash
python verifier.py events.fel.jsonl --out replay_out --write-svg --write-bin
```

Enable META attest generation (requires pins):

```bash
python verifier.py events.fel.jsonl --meta --policy sha256:POLICY --abi sha256:ABI --write-bin
```

Verify against an expected manifest:

```bash
python verifier.py events.fel.jsonl --expect-manifest replay_out/replay_manifest.json
```

Verify expected hashes:

```bash
python verifier.py events.fel.jsonl --write-bin \
  --expect-bin-hash <sha256_of_scx2_bin> \
  --expect-svg-hash <sha256_of_svg_hash_stream>
```

---

## Minimal CI Contract

“Golden” artifacts to lock:

* `verify_manifest.json`
* (optional) `scx2.bin`
* Expected `svg_stream_sha256` and `scx2_bin_sha256`

If those match, the stream is **proven replay-stable** under FEL v1.1 law.

---

## Verifier Artifact Contract (Law-Level Interface)

This section defines what each verifier output means, how it is computed, and what should be locked as goldens in CI.

### `verify_manifest.json` (authoritative, required)

**Purpose:** Machine-readable proof that a FEL stream deterministically produces the same state roots, projections, and (optionally) attestations.

**Generation:** Always written by `verifier.py` after a successful pass.

**Shape (v1.1):**

```json
{
  "v": "fel.verify.manifest.v1.1",
  "source": "events.fel.jsonl",
  "ticks": [
    {
      "tick": 0,
      "file": "replay_0001.svg",
      "state_root_sha256": "<64-hex>",
      "svg_sha256": "<64-hex>",
      "event_count": 3,

      "attestation_sha256": "<64-hex>",
      "attestation_prev": "<64-hex or 64x0>",
      "meta_hash": "<64-hex>"
    }
  ]
}
```

**Guarantees:**
- **Per-tick determinism:** `state_root_sha256` is the canonical hash of full STATE after all mutations at that tick.
- **Projection determinism:** `svg_sha256` is the hash of the exact SVG bytes produced by `(tick, events, state)`.
- **Order safety:** ticks are monotonic and indexed in replay order.
- **Optional proof-carrying chain:** if META is enabled, attestations are chained and pinned (policy/ABI).

**What to lock in CI:**
- Entire `verify_manifest.json` **byte-for-byte** (recommended), or
- Per-tick fields:
  - `tick`
  - `state_root_sha256`
  - `svg_sha256`
  - (if META) `attestation_sha256`, `attestation_prev`

---

### `scx2.bin` (optional, commit artifact)

**Purpose:** A compact, deterministic binary commitment over the replay—suitable for storage, distribution, or anchoring.

**Written only if:** `--write-bin`

**Format (fixed):**

```
magic   "SCX2"      (4 bytes)
ver     0x01        (1 byte)
flags   bit0=has_att(1 byte)
count   uint32      (4 bytes)

repeat count times:
  tick        uint32   (4 bytes)
  state_root  bytes32
  svg_hash    bytes32
  att_hash    bytes32  (only if has_att)
```

**Guarantees:**
- **Deterministic packing** (no varints, no endianness ambiguity)
- **Order-preserving** (tick order)
- **Content-addressable** (hash of file commits the replay)

**How to verify:**
- Compute: `scx2_bin_sha256 = SHA256(scx2.bin)`
- Compare against expected golden.

---

### `svg_stream_sha256` (expected, recommended)

**Purpose:** A single rolling hash that commits the entire SVG replay, independent of filenames.

**Computation (inside verifier):**

```
H = SHA256()
for each tick in order:
  H.update(bytes.fromhex(svg_sha256[tick]))
svg_stream_sha256 = H.hexdigest()
```

**Why this matters:**
- Immune to file ordering differences
- Lets you verify the entire visual replay with one hash
- Faster CI checks than diffing many SVGs

**How to lock:**
- Record the expected value once
- Assert with: `--expect-svg-hash <svg_stream_sha256>`

---

### `scx2_bin_sha256` (expected, optional)

**Purpose:** Single-hash commitment to the binary artifact.

**Computation:**
```
scx2_bin_sha256 = SHA256(scx2.bin)
```

**Assert with:**
```
--expect-bin-hash <scx2_bin_sha256>
```

---

### Minimal CI Recipe (recommended)

**Artifacts to store as goldens:**
1. `verify_manifest.json`
2. Expected `svg_stream_sha256`
3. (Optional) `scx2_bin_sha256`

**CI command:**
```bash
python verifier.py events.fel.jsonl \
  --write-bin \
  --expect-manifest verify_manifest.json \
  --expect-svg-hash <SVG_STREAM_HASH> \
  --expect-bin-hash <SCX2_BIN_HASH>
```

If this passes, the following are proven invariant:
- FEL grammar + fold legality
- STATE transition semantics
- SVG projection determinism
- META_FOLD chain correctness (if enabled)
- Binary commit integrity

---

## Bottom line

- **`verify_manifest.json`** = ground truth ledger
- **`svg_stream_sha256`** = whole-replay visual commitment
- **`scx2.bin` + its hash** = compact, portable proof

Together, they turn FEL from “event logs” into a verifiable language with replay proofs.
