# CM1_FOLD_TENSOR.s7 — Formal Specification

**Artifact Class:** `SCXQ7::MicronautFoldTensor`
**Short Name:** `CM1_FOLD_TENSOR.s7`
**Version:** 1.0.0
**Status:** Frozen
**Authority:** `CONTROL_FOLD`
**Mutation:** forbidden

---

## 1. Purpose

`CM1_FOLD_TENSOR.s7` is the **canonical contracted tensor state** of the Micronaut
Fold System. It is a single sealed, deterministic binary artifact encoding:

- The complete 15-fold tensor space (axes, lane bindings, collapse rules)
- All 9 micronaut registrations (role, fold, authority)
- The SCXQ2 lane–fold mapping table
- The CM-1 control stream (C0 byte phase sequence)
- The SMCA DFA authority automaton (S0 → S5 + S⊥)
- The Merkle root over all lanes
- A global proof hash (SHA256)

It is **not** a runtime object. It is the system in contracted tensor form,
verifiable by the verifier at any time.

Formal identity:

```
Ψ_system = Σ (Fold_i ⊗ Block_j ⊗ State_k)
           with CM1_GATE enforcing S0 → S5 DFA
           sealed by SCXQ2 lanes
           Merkle-rooted
           WASM-verifiable
```

---

## 2. File Extension

`.s7` denotes a sealed SCXQ7-class object.

| Extension  | Meaning                                     |
|------------|---------------------------------------------|
| `.json`    | Editable document (NOT this)                |
| `.bin`     | Opaque blob (NOT this)                      |
| `.scx`     | Instruction stream (NOT this)               |
| `.tensor`  | ML weight file (NOT this)                   |
| `.s7`      | Sealed SCXQ7 class object (THIS)            |

---

## 3. Binary Layout

```
+--------------------------------------------------+
| SCXQ7 HEADER (40 bytes)                          |
|  0x00 [4B]  MAGIC                                |
|  0x04 [2B]  VERSION                              |
|  0x06 [1B]  CLASS                                |
|  0x07 [1B]  FLAGS                                |
|  0x08 [32B] ROOT_MERKLE_HASH                     |
+--------------------------------------------------+
| SCXQ2 LANE STREAM (variable)                     |
|  Per lane: [1B LANE_ID][4B LANE_LEN][payload]   |
+--------------------------------------------------+
| GLOBAL PROOF HASH (32 bytes)                     |
+--------------------------------------------------+
```

All multi-byte integer fields are **big-endian**.

---

## 4. Header Specification

### 4.1 Field Table

| Offset | Size | Type    | Field             | Value / Description                    |
|--------|------|---------|-------------------|----------------------------------------|
| 0x00   | 4B   | uint32  | MAGIC             | `0x53433737` ("SC77")                  |
| 0x04   | 2B   | uint16  | VERSION           | `0x0001`                               |
| 0x06   | 1B   | uint8   | CLASS             | `0x01` (MicronautFoldTensor)           |
| 0x07   | 1B   | uint8   | FLAGS             | Bitmask (see §4.2)                     |
| 0x08   | 32B  | bytes   | ROOT_MERKLE_HASH  | SHA256 Merkle root over all 5 lanes    |

Total header size: **40 bytes**.

### 4.2 FLAGS Bitmask

| Bit | Mask   | Name              | Meaning                                        |
|-----|--------|-------------------|------------------------------------------------|
| 0   | `0x01` | FROZEN            | Artifact is sealed; no mutation permitted      |
| 1   | `0x02` | MERKLE_VERIFIED   | Root Merkle hash is embedded and pre-verified  |
| 2   | `0x04` | CM1_EMBEDDED      | CM-1 control stream is present in EDGE lane    |

Current canonical value: `0x07` (all three bits set).

---

## 5. Lane Stream

Each lane is encoded as:

```
[1B LANE_ID] [4B LANE_LEN (big-endian)] [LANE_LEN bytes of payload]
```

Payload is **canonical JSON** (V0: keys sorted lexicographically, minimal
separators, UTF-8 encoded). The lane order is fixed:

| Order | LANE_ID | Lane Name | Contents                                |
|-------|---------|-----------|-----------------------------------------|
| 0     | `0x00`  | DICT      | Fold definitions (15 folds)             |
| 1     | `0x01`  | FIELD     | Micronaut registrations (9 micronauts)  |
| 2     | `0x02`  | LANE      | Intent routing + CM-1 phase table       |
| 3     | `0x03`  | EDGE      | DFA state graph + CM-1 control stream   |
| 4     | `0x04`  | BATCH     | Verifier rules (V0–V7) + artifact identity |

Lane order is **fixed** (V5: fold-to-lane mapping is fixed; mismatches rejected).

---

## 6. Lane Payload Schemas

### 6.1 DICT Lane — Fold Definitions

Schema URI: `scxq7://cm1-fold-tensor/dict/v1`

```json
{
  "fold_count": 15,
  "fold_definitions": [
    {
      "fold_id":       "⟁CONTROL_FOLD⟁",
      "lane":          "EDGE",
      "collapse_rule": ["Resolve", "Gate", "Commit"],
      "cm1_phase":     "@control.header.begin",
      "cm1_code":      "U+0001",
      "description":   "..."
    },
    ...
  ],
  "schema": "scxq7://cm1-fold-tensor/dict/v1"
}
```

All 15 fold entries are required. Keys are sorted (V0).

### 6.2 FIELD Lane — Micronaut Registrations

Schema URI: `scxq7://cm1-fold-tensor/field/v1`

```json
{
  "micronaut_count": 9,
  "micronaut_registrations": [
    {
      "authority": "none",
      "domain":    "pre-semantic",
      "fold":      "⟁CONTROL_FOLD⟁",
      "id":        "CM-1",
      "name":      "ControlMicronaut",
      "role":      "phase_geometry"
    },
    ...
  ],
  "schema": "scxq7://cm1-fold-tensor/field/v1"
}
```

All 9 micronaut entries are required. `authority` must be `"none"` for all (law invariant).

### 6.3 LANE Lane — Intent Routing + CM-1 Phases

Schema URI: `scxq7://cm1-fold-tensor/lane/v1`

Contains:
- `intent_routing`: ngram routing table, 9 routes, fallback=XM-1
- `cm1_phases`: 5 C0 phase entries keyed by Unicode code point string (`"U+0000"` … `"U+0004"`)

### 6.4 EDGE Lane — DFA + CM-1 Control Stream

Schema URI: `scxq7://cm1-fold-tensor/edge/v1`

Contains:
- `cm1_stream`: `[0, 1, 2, 3, 4]` — the five C0 control bytes
- `cm1_stream_encoding`: `"C0_control_bytes"`
- `dfa_state_graph`: full SMCA automaton (states S0–S5, S⊥; transitions; invariants)
- `phase_hashes`: `{ "U+0000": SHA256(0x00), … "U+0004": SHA256(0x04) }`

CM-1 is embedded here. Compute/UI folds ignore it. Only CONTROL_FOLD interprets it.

### 6.5 BATCH Lane — Verifier Rules + Artifact Identity

Schema URI: `scxq7://cm1-fold-tensor/batch/v1`

Contains:
- `verifier_rules`: V0–V7, each with `enabled: true` and description
- `artifact_identity`: class, name, version, status, fold/micronaut counts
- `law`: the three-sovereignty doctrine
- `authority`: `"KUHUL_π"`
- `mutation`: `"forbidden"`

---

## 7. Merkle Root Construction

Leaf hashes are SHA256 of each lane's canonical JSON payload:

```
L0 = SHA256(DICT_bytes)
L1 = SHA256(FIELD_bytes)
L2 = SHA256(LANE_bytes)
L3 = SHA256(EDGE_bytes)
L4 = SHA256(BATCH_bytes)
```

The tree is padded to the next power of 2 (8 nodes) by duplicating the last leaf:

```
L5 = L6 = L7 = L4

Level 1:  N0 = SHA256(L0 || L1)
          N1 = SHA256(L2 || L3)
          N2 = SHA256(L4 || L5)
          N3 = SHA256(L6 || L7)

Level 2:  N4 = SHA256(N0 || N1)
          N5 = SHA256(N2 || N3)

Root:     R  = SHA256(N4 || N5)
```

The root is stored in `ROOT_MERKLE_HASH` (header offset 0x08).

---

## 8. Global Proof Hash

The 32-byte global proof hash is appended at the end of the file:

```
proof_input = ROOT_MERKLE_HASH (32B)
            || CLASS (1B, big-endian)
            || VERSION (2B, big-endian)
            || FOLD_COUNT (1B, big-endian)

GLOBAL_PROOF_HASH = SHA256(proof_input)
```

For version 1.0.0 with 15 folds:

```
proof_input = <32B root> || 0x01 || 0x00 0x01 || 0x0F
```

---

## 9. Verifier Integration

The artifact is verifiable by `src/cm1_fold_tensor_builder.py`:

```bash
# Build canonical artifact
python src/cm1_fold_tensor_builder.py

# Verify existing artifact matches canonical build (V6: replay determinism)
python src/cm1_fold_tensor_builder.py --verify
```

A PASS means: same inputs produced identical bytes — the artifact is deterministic
and has not been mutated.

---

## 10. Canonical Hashes (v1.0.0)

The following hashes are the canonical reference for v1.0.0.
Any deviation indicates mutation or corruption.

| Field            | Hash (SHA256)                                                       |
|------------------|---------------------------------------------------------------------|
| DICT lane        | `daa0212b5c96a155abe6bc8ff69b70b4ce06e5a25082a47f8f2b282da8b9e834` |
| FIELD lane       | `9e333e0f2e57639d98323661329c196f60a699ee22efa83d9201cfcea4522906` |
| LANE lane        | `97b724f5984d991c5a3aa56d3161c73ea0b8b59459af703ac9904bb324572050` |
| EDGE lane        | `1ab287ddd47bdfb873c9dd78fca462944faf8aa916795126ddc2408c7b84b535` |
| BATCH lane       | `940f7103beacacff0cc4f3e6ed40803ac8ca729aa17160e08f1581a110f45b14` |
| Merkle root      | `84ad72a32a0047cc459bb4ec626dc77ab5679036faa300ecfc44531cdb1a0d02` |
| Global proof     | `bd4c13e9607b40d2bfff834ca434300a999c407f0d9969075f184c3e3e86a1f3` |

Total artifact size: **9274 bytes**.

---

## 11. Extension Points

The following extensions are defined but not yet implemented:

### 11.1 Append-Only Delta Extension

A `.s7d` delta file extends a frozen `.s7` by appending:

```
[4B DELTA_MAGIC = 0x53374400 ("S7D\x00")]
[2B PARENT_VERSION]
[32B PARENT_PROOF_HASH]
[SCXQ2 LANE STREAM (changed lanes only)]
[32B DELTA_PROOF_HASH]
```

The base artifact is never modified. Deltas chain via `PARENT_PROOF_HASH`.

### 11.2 WebRTC Streaming

`.s7` artifacts may be streamed lane-by-lane over WebRTC data channels:

- Each lane block `[1B ID][4B LEN][payload]` is a self-contained data channel message.
- The receiver verifies the Merkle root after receiving all 5 lanes.
- Partial delivery (lanes 0–3 without BATCH) is permitted for low-latency control plane bootstrap.

---

## 12. Authority + Mutation

```
Authority : KUHUL_π (sole execution authority)
Mutation  : forbidden
Branching : impossible
```

The artifact is sealed. The builder generates it deterministically from the
fold law declared in `micronaut/folds.toml`. To change the artifact, update
`folds.toml` and rebuild — the new hash becomes the new canonical reference.
