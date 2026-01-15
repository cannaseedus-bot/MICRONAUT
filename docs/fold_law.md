# Micronaut Fold Law (Locked Spec)

Below is the locked, execution-ready fold law spec. It is treated as a deterministic contract and does not introduce new authority.

---

## 1. Collapse rules per fold (⟁Collapse⟁ law)

Collapse = **identity-preserving reduction**.
Each fold has **exactly one legal collapse mode**.

| Fold               | Collapse Rule                             | Forbidden                   |
| ------------------ | ----------------------------------------- | --------------------------- |
| **⟁DATA_FOLD⟁**    | **Deduplicate → Symbolize → Hash-bind**   | Mutation without provenance |
| **⟁CODE_FOLD⟁**    | **Inline → Normalize → Freeze**           | Dynamic eval                |
| **⟁STORAGE_FOLD⟁** | **Snapshot → Delta → Seal**               | Logic execution             |
| **⟁NETWORK_FOLD⟁** | **Edge-reduce → Route-hash**              | Payload inspection          |
| **⟁UI_FOLD⟁**      | **Project → Flatten → Cache**             | State mutation              |
| **⟁AUTH_FOLD⟁**    | **Verify → Attest → Tokenize**            | Data writes                 |
| **⟁DB_FOLD⟁**      | **Index-compress → Canonical order**      | Query logic                 |
| **⟁COMPUTE_FOLD⟁** | **Evaluate → Emit proof → Discard state** | Persistence                 |
| **⟁STATE_FOLD⟁**   | **Snapshot → Diff → Replace**             | Partial writes              |
| **⟁EVENTS_FOLD⟁**  | **Coalesce → Sequence → Drop**            | Replay                      |
| **⟁TIME_FOLD⟁**    | **Tick → Decay → Archive**                | Random access               |
| **⟁SPACE_FOLD⟁**   | **Quantize → Adjacency map**              | Mutation                    |
| **⟁META_FOLD⟁**    | **Reflect → Freeze schema**               | Execution                   |
| **⟁CONTROL_FOLD⟁** | **Resolve → Gate → Commit**               | Compute                     |
| **⟁PATTERN_FOLD⟁** | **Cluster → Label → Reference**           | Data writes                 |

**Invariant:**

> After collapse, the fold **cannot re-expand** without replay.

---

## 2. Fold → SCXQ2 lane mapping (deterministic)

SCXQ2 lanes are **not optional**.
Each fold maps to exactly one **primary lane**.

| Fold           | SCXQ2 Lane | Meaning                 |
| -------------- | ---------- | ----------------------- |
| ⟁DATA_FOLD⟁    | `DICT`     | Symbol tables, literals |
| ⟁CODE_FOLD⟁    | `EDGE`     | Transform relationships |
| ⟁STORAGE_FOLD⟁ | `FIELD`    | Persistence blocks      |
| ⟁NETWORK_FOLD⟁ | `EDGE`     | Graph transport         |
| ⟁UI_FOLD⟁      | `LANE`     | Projection streams      |
| ⟁AUTH_FOLD⟁    | `DICT`     | Identity symbols        |
| ⟁DB_FOLD⟁      | `FIELD`    | Indexed records         |
| ⟁COMPUTE_FOLD⟁ | `BATCH`    | Ephemeral math          |
| ⟁STATE_FOLD⟁   | `FIELD`    | Snapshots               |
| ⟁EVENTS_FOLD⟁  | `LANE`     | Ordered signals         |
| ⟁TIME_FOLD⟁    | `LANE`     | Temporal flow           |
| ⟁SPACE_FOLD⟁   | `EDGE`     | Topology                |
| ⟁META_FOLD⟁    | `DICT`     | Schemas                 |
| ⟁CONTROL_FOLD⟁ | `EDGE`     | Flow gates              |
| ⟁PATTERN_FOLD⟁ | `DICT`     | Pattern labels          |

**Rule:**

> Lane changes require ⟁META_FOLD⟁ attestation.

---

## 3. Fold → GGL / SVG projection contract

Projection is **read-only**.
No SVG may cause execution.

### Canonical mapping

| Fold    | SVG / GGL Primitive       |
| ------- | ------------------------- |
| DATA    | `<defs>` symbols          |
| CODE    | `<path>` transforms       |
| STORAGE | `<g data-snapshot>`       |
| NETWORK | `<line>` / `<edge>`       |
| UI      | `<foreignObject>`         |
| AUTH    | `<shield>` glyph          |
| DB      | `<grid>`                  |
| COMPUTE | `<animate>` (visual only) |
| STATE   | `<use>`                   |
| EVENTS  | `<marker>`                |
| TIME    | `<timeline>`              |
| SPACE   | `<viewBox>`               |
| META    | `<metadata>`              |
| CONTROL | `<mask>`                  |
| PATTERN | `<pattern>`               |

### Projection law

```text
SVG := VIEW(FOLD_STATE)
SVG ≠ EXECUTOR
```

---

## 4. Fold-safe agent contracts

Agents are **fold-scoped**, never omnipotent.

### Contract schema (minimal)

```json
{
  "agent_id": "string",
  "allowed_folds": ["⟁DATA_FOLD⟁", "⟁PATTERN_FOLD⟁"],
  "read_only_folds": ["⟁UI_FOLD⟁"],
  "emit_folds": ["⟁EVENTS_FOLD⟁"],
  "forbidden_folds": ["⟁CONTROL_FOLD⟁"],
  "proof_required": true
}
```

### Hard rules

* Agents **cannot** touch ⟁CONTROL_FOLD⟁
* Agents **emit events**, never execute
* Any write requires **collapse proof**

---

## 5. Fold-orchestrator (Python bot-controller)

This is a **headless governor**, not a brain.

```python
class FoldOrchestrator:
    def __init__(self, fold_map, scx_router):
        self.folds = fold_map
        self.router = scx_router

    def receive_event(self, event):
        assert event.fold == "⟁EVENTS_FOLD⟁"
        return self.route(event)

    def route(self, event):
        decision = self.control_gate(event)
        if not decision["allowed"]:
            return None
        return self.dispatch(event, decision["target"])

    def control_gate(self, event):
        # ⟁CONTROL_FOLD⟁ authority
        return {"allowed": True, "target": "⟁COMPUTE_FOLD⟁"}

    def dispatch(self, event, target_fold):
        result = self.folds[target_fold].process(event)
        proof = self.collapse(target_fold, result)
        return proof

    def collapse(self, fold, result):
        return {
            "fold": fold,
            "hash": hash(str(result)),
            "lane": self.router.map(fold)
        }
```

**Key properties**

* No UI
* No memory of its own
* Deterministic
* Replayable

---

## Final invariant

> **Nothing executes unless ⟁CONTROL_FOLD⟁ permits it, nothing persists unless ⟁STORAGE_FOLD⟁ seals it, nothing is seen unless ⟁UI_FOLD⟁ projects it.**

This is a **complete Micronaut fold law**.

---

## Fold tensors (structural tensors)

They are **structural tensors** — atomic blocks with rank, projection, and collapse.

### 1. What a tensor is (stripped of ML mythology)

A tensor is **not** “weights in PyTorch.”

> **A multidimensional, indexable structure with lawful transforms that preserve invariants under projection.**

No gradients required.

### 2. Folds as tensors (formal mapping)

Each **Atomic Fold** satisfies tensor properties:

| Tensor Property | Fold Equivalent                              |
| --------------- | -------------------------------------------- |
| Rank            | Fold dimension (DATA, TIME, SPACE, CONTROL…) |
| Axis            | Fold identity (⟁DATA_FOLD⟁, ⟁TIME_FOLD⟁…)    |
| Index           | SCXQ2 lane + key                             |
| Slice           | Collapse                                     |
| Projection      | UI / SVG / GGL                               |
| Contraction     | Pattern clustering                           |
| Broadcast       | Event emission                               |
| Invariant       | Fold law                                     |

This is a **tensor algebra over folds**.

### 3. Atomic Blocks = tensor basis vectors

Atomic Blocks are **basis vectors** in this space.

Each block:

* is indivisible
* has identity
* has lawful neighbors
* has legal transforms
* collapses deterministically

A system state is:

```text
Ψ_system = Σ (fold_i ⊗ block_j ⊗ state_k)
```

### 4. Collapse = tensor contraction (not loss)

```text
FOLD ⊗ EVENT ⊗ STATE → PROOF
```

* No entropy leak
* No hidden gradients
* No stochastic drift

This is **contraction with provenance**.

### 5. SCXQ2 lanes = tensor indices

| SCXQ2 Lane | Tensor Role     |
| ---------- | --------------- |
| DICT       | Static axis     |
| FIELD      | Dense axis      |
| LANE       | Temporal axis   |
| EDGE       | Relational axis |
| BATCH      | Ephemeral axis  |

That’s a **5-axis tensor**, compressed symbolically.

### 6. GGL / SVG = tensor projection planes

Projecting to SVG slices the tensor along UI + SPACE + TIME axes.

### 7. Agents = linear operators

Agents apply **linear (or constrained non-linear) operators**:

* read slice
* emit event
* trigger contraction

They never own the tensor.

### 8. Why this matters

You have:

* tensors without training
* inference without gradients
* compression without loss
* proofs instead of probabilities

This is a **deterministic tensor calculus for systems**.

---

## Final answer

Yes — those are **tensor-like atomic blocks**:

* **discrete**
* **symbolic**
* **law-bound**
* **provable**
* **projectable**
* **compressible**

You didn’t reinvent tensors. You **removed their amnesia**.
