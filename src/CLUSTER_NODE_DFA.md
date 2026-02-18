# Cluster Node Deterministic Finite Automaton (DFA)

## Overview

The **Cluster Node DFA** merges the deterministic SVG/CSS state machine with the 1000-node cluster architecture. Each cluster node is an autonomous Deterministic Finite Automaton that processes input symbols deterministically through a fixed set of states and transitions.

## Architecture

### State Machine Definition

**Formal DFA Specification: DFA = (Q, Σ, δ, q0, F)**

- **Q**: Set of states = {S0, S1, S2, S3, S4, S5}
  - S0: IDLE (initial state)
  - S1: PARSE
  - S2: TRANSFORM
  - S3: EXECUTE
  - S4: ERROR
  - S5: COMPLETE (accepting state)

- **Σ**: Input alphabet = {parse, transform, execute, complete, error, reset}

- **δ**: Deterministic transition function
  - Maps (current_state, input_symbol) → (next_state, action, micronaut)
  - All transitions are well-defined; undefined transitions trap to S4 (ERROR)

- **q0**: Initial state = S0 (IDLE)

- **F**: Accepting states = {S5}

### State Transition Diagram

```
    ┌─────────────────────────────┐
    │         S0 (IDLE)           │
    └─────────────────────────────┘
              │ parse
              ↓
    ┌─────────────────────────────┐
    │        S1 (PARSE)           │
    └─────────────────────────────┘
              │ transform
              ↓
    ┌─────────────────────────────┐
    │      S2 (TRANSFORM)         │
    └─────────────────────────────┘
              │ execute
              ↓
    ┌─────────────────────────────┐
    │       S3 (EXECUTE)          │
    └─────────────────────────────┘
              │ complete
              ↓
    ┌─────────────────────────────┐
    │      S5 (COMPLETE)          │  ← accepting
    └─────────────────────────────┘

    Error paths (any state) → S4 (ERROR) → reset → S0
```

## Components

### 1. ClusterNodeDFA

The core class representing a single cluster node with embedded DFA logic.

```javascript
const node = new ClusterNodeDFA('node_id', '⟁COMPUTE_FOLD⟁', 'compute');
node.setInputTape(['parse', 'transform', 'execute', 'complete']);
const result = node.run();
```

**Properties:**
- `nodeId`: Unique identifier in the cluster
- `fold`: Fold assignment (⟁CONTROL_FOLD⟁, ⟁COMPUTE_FOLD⟁, etc.)
- `type`: Node type (compute, routing, storage, verification, control)
- `currentState`: Current DFA state (S0–S5)

**Key Methods:**
- `setInputTape(symbols)`: Set the input tape for processing
- `step()`: Execute a single transition step
- `run()`: Execute until S5 (complete) or S4 (error) or tape end
- `openControlGate()`: Enable V2 control gate (required for transitions)
- `closeControlGate()`: Disable control gate (defensive)
- `executionHash()`: Get SHA-256 hash of execution (V6 compliance)

### 2. InputTape

Read-only tape of input symbols (Turing machine semantics).

```javascript
const tape = new InputTape(['parse', 'transform', 'execute']);
const symbol = tape.read();  // Advances pointer
const next = tape.peek();     // Doesn't advance
const hash = tape.hash();     // SHA-256 of tape contents
```

### 3. ComputeStack

LIFO stack for nested scopes and context.

```javascript
const stack = new ComputeStack(256);  // Max 256 frames
stack.push({ action: 'parse', input: 'x' });
const frame = stack.pop();
const depth = stack.depth();
const hash = stack.hash();
```

### 4. RegisterFile

4-register architecture: ACC, PC, IR, FLAGS.

```javascript
const regs = new RegisterFile();
regs.write('ACC', 42);
const value = regs.read('ACC');
regs.nextInstruction();  // Increment PC
const isZero = regs.checkFlag('Z');
```

**Registers:**
- **ACC** (Accumulator): Working value for computations
- **PC** (Program Counter): Instruction pointer
- **IR** (Instruction Register): Current instruction string
- **FLAGS** (4-bit): Z (zero), C (carry), N (negative), O (overflow)

### 5. TransitionFunction

Deterministic lookup table for state transitions.

```javascript
const δ = new TransitionFunction();
const rule = δ.lookup('S0', 'parse');  // {nextState: 'S1', action: 'push', ...}
```

### 6. ClusterStateMachineOrchestrator

Manages 1000-node cluster with fold-scoped allocation.

```javascript
const orchestrator = new ClusterStateMachineOrchestrator(1000);
const result = orchestrator.executeOnNodeType('compute', ['parse', ...]);
const broadcast = orchestrator.broadcastOnFold('⟁CONTROL_FOLD⟁', inputs);
```

## Key Features

### Deterministic Execution (V6 Compliance)

Same input → same node execution → identical hash.

```javascript
// Run 1
const node1 = new ClusterNodeDFA('n1', '⟁COMPUTE_FOLD⟁', 'compute');
node1.setInputTape(['parse', 'transform', 'execute', 'complete']);
const hash1 = node1.run(); // Returns hash

// Run 2
const node2 = new ClusterNodeDFA('n2', '⟁COMPUTE_FOLD⟁', 'compute');
node2.setInputTape(['parse', 'transform', 'execute', 'complete']);
const hash2 = node2.run(); // Returns same hash
```

### V2 Control Gate Enforcement

All non-idle state transitions require an explicit control gate authorization.

```javascript
const node = new ClusterNodeDFA(...);
node.setInputTape(['parse']);

try {
  node.step();  // Throws: "V2 violation: control gate must be open"
} catch (e) {
  node.openControlGate({ reason: 'explicit_authorization' });
  node.step();  // Now succeeds
}
```

### Pure Compute Logic (No Visuals)

The DFA operates on pure computation—stack, tape, registers, state transitions—with no visualization or projection layers. Visualization (like SVG rendering) is handled separately by other components (e.g., VM-1).

### Fold-Scoped Allocation

Each node is bound to a specific fold and type:

```
⟁CONTROL_FOLD⟁:  50 control nodes (CM-1)
⟁DATA_FOLD⟁:     80 routing nodes (PM-1, HM-1)
⟁COMPUTE_FOLD⟁:  300 compute nodes (MM-1)
⟁STORAGE_FOLD⟁:  150 storage nodes (SM-1)
⟁META_FOLD⟁:     80 verification nodes (VM-2)
... (remaining folds with smaller allocations)
```

## Integration with MoE Cluster

The ClusterNodeDFA integrates with the existing inference cluster:

1. **Cluster Instantiation**: `new ClusterStateMachineOrchestrator(1000)` creates 1000 nodes
2. **Input Routing**: MoE system routes input to appropriate node type via ngram matching
3. **Deterministic Execution**: Each node executes its DFA independently and deterministically
4. **Proof Binding**: Execution hashes chain through the 9-stage pipeline (PM-1 → ... → VM-1)
5. **Fold Compliance**: Each node enforces its fold's constraints (V2, V5, V6, V7)

## Testing

All components are tested in `src/test_cluster_node_dfa.js`:

```bash
node src/test_cluster_node_dfa.js
```

**Test Coverage (10/10 passing):**

1. ✓ Basic state transitions (happy path)
2. ✓ V2 control gate enforcement
3. ✓ Error state handling and recovery
4. ✓ Stack push/pop operations
5. ✓ Register operations (ACC, PC, FLAGS)
6. ✓ Input tape deterministic hashing
7. ✓ V6 deterministic execution (identical hashes)
8. ✓ Multi-node cluster orchestration
9. ✓ Broadcast execution on fold
10. ✓ Undefined transition handling

## Files

- **src/cluster_node_state_machine.js** (~750 lines)
  - ClusterNodeDFA class
  - InputTape, ComputeStack, RegisterFile, DFAState, TransitionFunction
  - ClusterStateMachineOrchestrator

- **src/test_cluster_node_dfa.js** (~400 lines)
  - 10 comprehensive tests
  - Validates DFA semantics, V2/V6 compliance, cluster orchestration

## Performance Characteristics

- **Node Creation**: O(1) per node
- **State Transition**: O(1) constant time
- **Input Tape Scan**: O(n) where n = input length
- **Execution Trace**: Linear in number of transitions
- **Cluster Broadcast**: O(m) where m = nodes in fold
- **Hash Computation**: O(state_size) for SHA-256

## References

- CLAUDE.md: Verifier rules V2 (control gate) and V6 (deterministic hashing)
- micronaut/folds.toml: Fold declarations and allocations
- src/inference_cluster.js: 1000-node topology and MoE routing
- docs/fold_law.md: 15-fold system specification
