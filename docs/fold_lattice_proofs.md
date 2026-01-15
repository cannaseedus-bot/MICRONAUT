# Fold lattice proofs

Treat fold access as a **partial order** over capability sets.

## Fold lattice

Let:

* **F** = set of folds
* **A(agent)** = allowed folds
* **R(agent)** = read-only folds
* **E(agent)** = emit-only folds (typically EVENTS)
* **W(agent)** = write folds (rare; usually DATA/STATE via COMPUTE gate)
* **X(agent)** = forbidden folds (hard deny)

Define the **capability lattice** as:

* Meet (⊓) = intersection of allowed capability sets
* Join (⊔) = union, only valid if **META attests** (cap escalation is illegal without proof)

### Minimal ordering

`read ⊑ emit ⊑ write ⊑ control`

With a hard ceiling:

* `CONTROL` is **not grantable** to agents.
* `AUTH` is verify-only.
* `UI` is read-only.

## Proof object (agent action proof)

Every non-trivial action emits a proof:

```json
{
  "proof_v": "fold.proof.v1",
  "agent_id": "agent://name",
  "action_id": "uuid",
  "event_hash": "blake3:...",
  "requested": {
    "reads": ["⟁STATE_FOLD⟁"],
    "emits": ["⟁EVENTS_FOLD⟁"],
    "writes": []
  },
  "lattice_check": {
    "allowed": true,
    "reason": "requested ⊑ contract"
  },
  "fold_effects": [
    { "fold": "⟁EVENTS_FOLD⟁", "effect": "emit", "hash": "blake3:..." }
  ],
  "lane_map": {
    "⟁EVENTS_FOLD⟁": "LANE"
  },
  "attest": {
    "meta_hash": "blake3:...",
    "policy_hash": "blake3:..."
  }
}
```

## Lattice proof rules (the core the verifier enforces)

### Rule L1 — Subset legality

For any request **req** and agent contract **ctr**:

* `req.reads ⊆ (ctr.allowed_folds ∪ ctr.read_only_folds)`
* `req.emits ⊆ ctr.emit_folds`
* `req.writes ⊆ ctr.allowed_folds` **AND** must be mediated by COMPUTE gate (see verifier)

### Rule L2 — No-control authority

`⟁CONTROL_FOLD⟁ ∉ (req.reads ∪ req.writes ∪ req.emits)` always.

### Rule L3 — UI is projection-only

`⟁UI_FOLD⟁ ∉ req.writes` always.

### Rule L4 — Writes must be “via compute”

If `req.writes` non-empty then the proof must include:

* a compute trace hash
* a collapse hash
* a storage seal hash if persisted

This is how you prevent “UI blob writes state”.
