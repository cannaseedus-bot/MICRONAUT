# SWMSP v1.0.0 ↔ S7-LLM Integration Mapping (Frozen v1)

Status: **Frozen**  
Mutation: **Forbidden without MAJOR protocol bump**  
Execution Authority: **Server-only inference**  
Storage Authority: **Merkle root (`root_announcement.merkle_root`)**

## 1) Protocol Contract

This document binds **SWMSP v1.0.0** (Service Worker Merkle Shard Protocol) to the S7 model family as a structural mapping. It does **not** grant client inference authority.

- Client/service-worker side: storage, transport, and proof validation only.
- Server side: routing, execution, and logit computation.

This is a distributed weight fabric, not distributed inference authority.

## 2) Frozen SWMSP JSON Schema (Authoritative)

```json
{
  "$id": "swmsp://schema/v1",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Service Worker Merkle Shard Protocol v1.0.0",
  "type": "object",
  "oneOf": [
    { "$ref": "#/$defs/root_announcement" },
    { "$ref": "#/$defs/shard_descriptor" },
    { "$ref": "#/$defs/shard_request" },
    { "$ref": "#/$defs/shard_response" },
    { "$ref": "#/$defs/verification_result" },
    { "$ref": "#/$defs/node_capability" }
  ],
  "$defs": {
    "hash_hex": {
      "type": "string",
      "pattern": "^[a-fA-F0-9]{64}$"
    },
    "model_id": {
      "type": "string",
      "minLength": 1
    },
    "dtype_enum": {
      "type": "string",
      "enum": ["int8", "int4", "fp16", "fp32"]
    },
    "root_announcement": {
      "type": "object",
      "required": [
        "type",
        "model_id",
        "protocol_version",
        "merkle_root",
        "total_shards",
        "shard_size_bytes"
      ],
      "properties": {
        "type": { "const": "root_announcement" },
        "protocol_version": { "const": "1.0.0" },
        "model_id": { "$ref": "#/$defs/model_id" },
        "merkle_root": { "$ref": "#/$defs/hash_hex" },
        "total_shards": { "type": "integer", "minimum": 1 },
        "shard_size_bytes": { "type": "integer", "minimum": 1 },
        "created_at": { "type": "integer" }
      },
      "additionalProperties": false
    },
    "shard_descriptor": {
      "type": "object",
      "required": [
        "type",
        "model_id",
        "layer_id",
        "tensor_id",
        "shard_index",
        "total_shards",
        "dtype",
        "shape",
        "chunk_hash"
      ],
      "properties": {
        "type": { "const": "shard_descriptor" },
        "model_id": { "$ref": "#/$defs/model_id" },
        "layer_id": { "type": "integer", "minimum": 0 },
        "tensor_id": { "type": "string" },
        "shard_index": { "type": "integer", "minimum": 0 },
        "total_shards": { "type": "integer", "minimum": 1 },
        "dtype": { "$ref": "#/$defs/dtype_enum" },
        "shape": {
          "type": "array",
          "items": { "type": "integer", "minimum": 1 },
          "minItems": 1
        },
        "chunk_hash": { "$ref": "#/$defs/hash_hex" }
      },
      "additionalProperties": false
    },
    "shard_request": {
      "type": "object",
      "required": ["type", "model_id", "layer_id", "tensor_id", "shard_index"],
      "properties": {
        "type": { "const": "shard_request" },
        "model_id": { "$ref": "#/$defs/model_id" },
        "layer_id": { "type": "integer", "minimum": 0 },
        "tensor_id": { "type": "string" },
        "shard_index": { "type": "integer", "minimum": 0 }
      },
      "additionalProperties": false
    },
    "merkle_proof": {
      "type": "object",
      "required": ["leaf_hash", "proof_path"],
      "properties": {
        "leaf_hash": { "$ref": "#/$defs/hash_hex" },
        "proof_path": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["position", "hash"],
            "properties": {
              "position": { "type": "string", "enum": ["left", "right"] },
              "hash": { "$ref": "#/$defs/hash_hex" }
            },
            "additionalProperties": false
          }
        }
      },
      "additionalProperties": false
    },
    "shard_response": {
      "type": "object",
      "required": [
        "type",
        "model_id",
        "layer_id",
        "tensor_id",
        "shard_index",
        "chunk_hash",
        "shard_bytes_base64",
        "merkle_proof"
      ],
      "properties": {
        "type": { "const": "shard_response" },
        "model_id": { "$ref": "#/$defs/model_id" },
        "layer_id": { "type": "integer", "minimum": 0 },
        "tensor_id": { "type": "string" },
        "shard_index": { "type": "integer", "minimum": 0 },
        "chunk_hash": { "$ref": "#/$defs/hash_hex" },
        "shard_bytes_base64": { "type": "string", "contentEncoding": "base64" },
        "merkle_proof": { "$ref": "#/$defs/merkle_proof" }
      },
      "additionalProperties": false
    },
    "verification_result": {
      "type": "object",
      "required": ["type", "model_id", "layer_id", "tensor_id", "shard_index", "verified"],
      "properties": {
        "type": { "const": "verification_result" },
        "model_id": { "$ref": "#/$defs/model_id" },
        "layer_id": { "type": "integer" },
        "tensor_id": { "type": "string" },
        "shard_index": { "type": "integer" },
        "verified": { "type": "boolean" },
        "computed_root": { "$ref": "#/$defs/hash_hex" }
      },
      "additionalProperties": false
    },
    "node_capability": {
      "type": "object",
      "required": ["type", "node_id", "storage_bytes", "bandwidth_kbps", "redundancy_factor"],
      "properties": {
        "type": { "const": "node_capability" },
        "node_id": { "type": "string" },
        "storage_bytes": { "type": "integer", "minimum": 0 },
        "bandwidth_kbps": { "type": "integer", "minimum": 0 },
        "redundancy_factor": { "type": "integer", "minimum": 1 }
      },
      "additionalProperties": false
    }
  }
}
```

## 3) Frozen Invariants (Authoritative)

1. `merkle_root` is immutable.
2. `chunk_hash` MUST equal `SHA-256(shard_bytes)`.
3. `leaf_hash` MUST equal `chunk_hash`.
4. Reconstructed proof root MUST equal `root_announcement.merkle_root`.
5. `additionalProperties: false` prevents silent extension.
6. Any structural schema change requires MAJOR protocol bump.

## 4) Deterministic Verification Rule

```text
computed_leaf = SHA256(decoded_shard_bytes)

assert computed_leaf == chunk_hash
assert computed_leaf == merkle_proof.leaf_hash

reconstructed_root = MerkleRebuild(leaf_hash, proof_path)

assert reconstructed_root == root_announcement.merkle_root

verified = true
```

If any assertion fails, reject the shard.

## 5) S7 Parameter Topology Mapping

Canonical tensor domains:

```text
Embedding
TransformerBlock[0..N-1]
  ├── Wq
  ├── Wk
  ├── Wv
  ├── Wo
  ├── FFN_W1
  ├── FFN_W2
MoE_Experts[0..E-1]
  ├── Expert_W1
  ├── Expert_W2
LM_Head
```

Tensor identity rule:

```text
tensor_id = "{layer}.{component}"
```

Examples:

- `embeddings.weight`
- `block.3.attn.Wq`
- `block.3.ffn.W1`
- `moe.expert.12.W1`
- `lm_head.weight`

### S7 → SWMSP Field Mapping

| S7 concept | SWMSP field |
|---|---|
| Model identifier | `model_id` |
| Layer index | `layer_id` |
| Tensor name | `tensor_id` |
| Data type | `dtype` |
| Tensor shape | `shape` |
| Shard ordinal | `shard_index` |
| Raw shard bytes | `shard_bytes_base64` |
| Chunk digest | `chunk_hash` |
| Global authority digest | `merkle_root` |

## 6) Deterministic Shard Derivation

Shard index and cryptographic binding are deterministic:

```text
shard_index = floor(byte_offset / shard_size_bytes)
chunk_hash  = SHA256(raw_tensor_chunk_bytes)
leaf_hash   = chunk_hash
```

Global root derives from all leaf hashes:

```text
merkle_root = MerkleTree(leaf_hashes)
```

Published in:

```text
root_announcement.merkle_root
```

## 7) MoE Expert Partitioning

Expert partition rule:

```text
Expert i := all tensors with prefix "moe.expert.i.*"
```

Optional subtree commitment per expert:

```text
expert_root_hash
```

Global composition may be represented as:

```text
global_root = Merkle(
  embedding_root,
  transformer_root,
  expert_root[0..E-1],
  lm_head_root
)
```

This permits selective expert loading and independent expert verification while preserving global-root authority.

## 8) Service Worker Allocation Model

Service Worker responsibilities:

1. Cache verified shards.
2. Serve `shard_response` payloads.
3. Validate Merkle proof before cache commit.
4. Maintain local shard-availability index.

Service Worker prohibitions:

- No logit computation.
- No tensor mutation.
- No routing policy rewrite.

Example local availability map:

```json
{
  "model_id": "s7-llm-140m",
  "local_shards": [
    { "layer_id": 3, "tensor_id": "block.3.attn.Wq", "shard_index": 2 },
    { "layer_id": 3, "tensor_id": "block.3.ffn.W1", "shard_index": 1 }
  ]
}
```

## 9) Inference-Time Shard Flow

1. Router selects active experts (e.g., `[12, 43]`).
2. Engine derives required tensor list from routing decision.
3. For each required shard:
   - use local cache if verified and present,
   - otherwise issue `shard_request`, verify proof, commit to cache.
4. Only verified tensors are deserialized/bound into runtime memory.

No proof → no tensor mapping.

## 10) FIELD Lane Binding for S7

For S7 integration, lane `id=2` (FIELD) carries sealed INT8 tensor payloads.

Logical payload layout:

```text
[Tensor Header]
[Shard Count]
[Shard Offsets]
[Shard Bytes]
```

Deterministic bind sequence:

1. Verify shard hash + Merkle proof.
2. Reconstruct tensor bytes.
3. Map bytes into aligned memory region.
4. Bind pointer/reference into inference engine.

## 11) Trust Model

| Component | Trust level |
|---|---|
| Service Worker | Zero trust |
| Browser cache | Zero trust |
| Client compute | Zero trust |
| Shard host peers | Zero trust |
| Root hash | Absolute authority |
| Verifier | Absolute authority |

Only verified shards may enter inference memory.

## 12) Determinism Guarantee

Given identical:

- `model_id`
- `merkle_root`
- routing decision
- inference seed/config

system output must be identical, even when shards are sourced from different peers.

```text
Same model_id
Same merkle_root
Same routing decision
Same seed
→ Identical logits
```

## 13) Scaling and Versioned Expansion

To scale parameter count:

1. Add/modify tensors.
2. Re-shard deterministically.
3. Recompute Merkle root.
4. Publish new `root_announcement`.

Clients scale by fetching/verifying newly required shards against the new root. No protocol mutation is required for this expansion path.

## 14) What This Enables

- Deterministic distributed weight storage
- Untrusted client shard hosting
- Centralized inference authority
- On-demand expert loading
- Versioned model lineage via root commitments

## 15) What This Does NOT Permit

- Client logit computation authority
- Client shard mutation authority
- Silent protocol extension
- Root rewriting
