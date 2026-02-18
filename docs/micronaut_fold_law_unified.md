# Unified Fold Algebra (Micronaut)

This document freezes the fold-only deterministic model into algebraic laws and verification artifacts.

## 1) Unified Fold Basis

```
F = { DATA, CODE, STORAGE, NETWORK, UI, AUTH,
      DB, COMPUTE, STATE, EVENTS, TIME,
      SPACE, META, CONTROL, PATTERN }
```

System state:

```
Ψ = Σ (F_i ⊗ S_i)
```

No fold mixes axes unless permitted by CONTROL.

## 2) Collapse as Contraction

Each fold defines one contraction operator:

```
C_F : (F ⊗ X) → Proof_F
```

Required properties:

1. Deterministic
2. Identity-preserving
3. Replayable
4. Non-expanding

Idempotence:

```
C_F(C_F(X)) = C_F(X)
```

## 3) Lane Mapping as Tensor Indexing

```
Index(F) = Lane
```

SCXQ2 lane roles:

- `DICT`: static symbolic axes
- `FIELD`: dense stored axes
- `LANE`: temporal streams
- `EDGE`: relational graph axes
- `BATCH`: ephemeral compute axis

Full tensor form:

```
T[dict, field, lane, edge, batch]
```

Lane reassignment requires META attestation:

```
Index(F_old) ≠ Index(F_new) → must pass C_META
```

## 4) Projection Law

Projection is read-only:

```
P : Ψ → VIEW
```

Constraint:

```
P ∘ C_F = C_F ∘ P
```

So:

```
SVG = P(Ψ)
Ψ ≠ SVG
```

## 5) Agents as Bounded Operators

```
A : Slice(Ψ) → Event
A ∈ L_bounded(F_subset)
```

Agent constraints:

- no CONTROL operator
- no direct STORAGE mutation
- emits via EVENTS

## 6) CONTROL as Gate Matrix

```
G : Event → {allowed, target_fold}
```

Execution condition:

```
Execute ⇔ CONTROL(Event) = Permit
```

## 7) Persistence Law

Persistence requires storage collapse:

```
C_STORAGE(result)
```

Compute without storage collapse leaves no residual state.

## 8) TIME as Forward-Only Axis

```
T[n] = f(T[n-1])
```

Historical random reads are forbidden and must be reconstructed by replay.

## 9) PATTERN as Second-Order Contraction

```
Cluster(Ψ_subset) → Label
```

Labels are stored in DICT. PATTERN remains axis-distinct.

## 10) Minimal Fold Grammar

```
<system> ::= <event>*
<event> ::= "(" FOLD ":" payload ")"
<payload> ::= symbol | structure | reference
<transition> ::= EVENTS -> CONTROL -> TARGET_FOLD -> COLLAPSE
<collapse> ::= HASH "[" canonical_form "]"
<projection> ::= VIEW "(" FOLD_STATE ")"
```

No implicit mutation, no hidden state.

## 11) Unified Cycle Equation

```
Ψ_next = Σ C_F ( G( A( Slice(Ψ_current) ) ) )
```

## Frozen Artifacts

- Frozen schema: `docs/micronaut-fold-law.schema.json`
- Deterministic verifier: `src/fold_verifier.py`

Minimal verifier checks:

1. Schema-level structural conformance
2. Completeness of all 15 folds
3. Lane validity and deterministic mapping keys
4. Collapse idempotence via canonical hash stability

Execution cycle target:

```
EVENTS → CONTROL → TARGET → COLLAPSE → HASH
```

Verifier invariant:

```
HASH(system) is stable
```
