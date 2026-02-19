# SWMSP v1.0.0 ↔ S7-LLM Integration Mapping (Frozen v1)

Status: **Frozen**  
Mutation: **Forbidden without MAJOR protocol bump**  
Execution Authority: **Server-only inference**  
Storage Authority: **Merkle root (`root_announcement.merkle_root`)**

## 1) Protocol Contract

This document binds **SWMSP v1.0.0** (Service Worker Merkle Shard Protocol) to the S7 model family as a structural mapping. It does **not** grant client inference authority.

- Client/service-worker side: storage, transport, and proof validation only.
- Server side: routing, execution, and logit computation.

## 2) Frozen Message Schema Surface

SWMSP v1.0.0 message families:

- `root_announcement`
- `shard_descriptor`
- `shard_request`
- `shard_response`
- `verification_result`
- `node_capability`

All objects enforce `additionalProperties: false`; silent extensions are forbidden in v1.0.0.

## 3) Frozen Invariants (Authoritative)

1. `merkle_root` is immutable for a published `model_id` + `protocol_version` bundle.
2. `chunk_hash` MUST equal `SHA-256(shard_bytes)`.
3. `leaf_hash` MUST equal `chunk_hash`.
4. Reconstructed proof root MUST equal `root_announcement.merkle_root`.
5. Structural schema changes require MAJOR protocol bump.
6. Verification failure means shard rejection (never mapped into inference memory).

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

the system must produce identical logits, even when shards are sourced from different peers.

## 13) Scaling and Versioned Expansion

To scale parameter count:

1. Add/modify tensors.
2. Re-shard deterministically.
3. Recompute Merkle root.
4. Publish new `root_announcement`.

Clients scale by fetching/verifying newly required shards against the new root. No protocol mutation is required for this expansion path.
