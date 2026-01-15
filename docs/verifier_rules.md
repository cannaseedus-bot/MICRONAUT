# Formal verifier rules (the “no drift” contract)

These are the minimum rules that make the whole universe deterministic.

## V0 — Canonical ordering

All maps/objects are verified in canonical order:

* keys sorted lexicographically
* arrays preserve order
* numeric types must be normalized (no NaN/Inf)

## V1 — Fold legality

Every operation references exactly one fold in `op.fold`, and:

* fold must be in COMPLETE_FOLD_MAP
* fold access must satisfy lattice rules (see `docs/fold_lattice_proofs.md`)

## V2 — Control gate monopoly

Any state mutation requires an explicit control gate record:

* `control.decide_hash`
* `control.policy_hash`
* `control.target_fold`
* `control.allow = true`

No “implicit state writes”.

## V3 — Compute mediation

If DATA or STATE changes, the proof must contain:

* `compute.trace_hash`
* `collapse.hash`
* optional `storage.seal_hash` if persisted

## V4 — UI read-only

Any op targeting UI must be tagged `projection=true` and:

* must not write to DATA/STATE/STORAGE
* may only read: DATA/STATE/TIME/SPACE/PATTERN/META (policy-defined)

## V5 — Lane binding

Fold → lane must match the fixed map.
If a proof says `⟁EVENTS_FOLD⟁ : FIELD` the verifier rejects.

## V6 — Replay determinism

Given the same:

* input events
* fold map
* policy hash
* lane map

…the verifier must reproduce the same:

* collapse hashes
* snapshot hashes
* packed binary stream

## V7 — Hash binding (ABI drift stopper)

All proofs include:

* `abi_hash` (schema + fold rules + lane map + canonicalization)
* `policy_hash`
* `meta_hash`

If any changes, it is a version bump (MAJOR/MINOR per ASX-R rules).
