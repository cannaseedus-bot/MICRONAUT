# SMCA Authority Automaton (DFA)

## States

```
S0 = MATRIX_PROPOSAL
S1 = CM1_GATE
S2 = SCXQ7_LAW
S3 = SCXQ2_SEMANTIC
S4 = SCO_EXECUTION
S5 = IDB_COMMIT
S⊥ = ILLEGAL
```

## Alphabet (Events)

```
propose
accept
reject
legal
illegal
execute
commit
```

## Transition Table

| Current | Event   | Next |
| ------- | ------- | ---- |
| S0      | propose | S1   |
| S1      | reject  | S⊥   |
| S1      | accept  | S2   |
| S2      | illegal | S⊥   |
| S2      | legal   | S3   |
| S3      | execute | S4   |
| S4      | commit  | S5   |
| S5      | *       | S5   |

## Invariants (Hard)

1. No transition skips a state.
2. No transition goes upward.
3. S⊥ is absorbing.
4. S5 is append-only.

## CM-1 Gate (Sub-Automaton)

```
INPUT_STREAM
   ↓
[balanced?] ──no──▶ ILLEGAL
   │
  yes
   ▼
[non-rendering?] ──no──▶ ILLEGAL
   │
  yes
   ▼
ACCEPT
```

CM-1 is purely rejecting, never transforming.

## Binary Splitting Legality (Derived)

Binary splitting is legal iff:

* collapse_class = binary.split
* tree is balanced
* no side effects
* recombination associative + deterministic

Otherwise → S⊥.
