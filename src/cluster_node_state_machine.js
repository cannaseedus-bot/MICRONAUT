/**
 * CLUSTER NODE STATE MACHINE — DFA Logic Embedded in Cluster Nodes
 *
 * Merges deterministic SVG/CSS state machine with 1000-node cluster architecture.
 * Each cluster node is a Deterministic Finite Automaton (DFA) with:
 *   - 6 states: S0(IDLE) → S1(PARSE) → S2(TRANSFORM) → S3(EXECUTE) → S5(COMPLETE)
 *                   ↓ error path → S4(ERROR) → reset → S0
 *   - Deterministic transition function δ: (State, Input) → (State, Action)
 *   - Pure compute logic: stack, tape, registers, program counter, accumulator
 *   - Deterministic hashing for V6 proof-binding compliance
 *
 * Architecture: No visuals, no projection—pure compute semantics only.
 * Each node's state machine is independent; orchestration happens at cluster level.
 */

import crypto from 'crypto';

/**
 * Deterministic Finite Automaton State Definition
 */
class DFAState {
  constructor(id, name, type = 'normal', color = '#16f2aa') {
    this.id = id;           // S0, S1, S2, S3, S4, S5
    this.name = name;       // IDLE, PARSE, TRANSFORM, EXECUTE, ERROR, COMPLETE
    this.type = type;       // 'initial' | 'normal' | 'error' | 'accepting'
    this.color = color;     // For visualization (unused in pure compute)
    this.entryTime = null;
    this.exitTime = null;
    this.transitionCount = 0;
  }

  enter(timestamp = Date.now()) {
    this.entryTime = timestamp;
  }

  exit(timestamp = Date.now()) {
    this.exitTime = timestamp;
    this.transitionCount += 1;
  }

  duration() {
    if (!this.entryTime || !this.exitTime) return 0;
    return this.exitTime - this.entryTime;
  }
}

/**
 * Deterministic Transition Function δ
 * Defines the legal state transitions based on input symbols
 */
class TransitionFunction {
  constructor() {
    // δ: (State, Input) → (State, Action)
    this.transitions = {
      'S0:parse':       { nextState: 'S1', action: 'push',     micronaut: 'PM-1' },
      'S0:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },
      'S0:reset':       { nextState: 'S0', action: 'idle',     micronaut: 'CM-1' },

      'S1:transform':   { nextState: 'S2', action: 'rewrite',  micronaut: 'TM-1' },
      'S1:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },
      'S1:reset':       { nextState: 'S0', action: 'pop',      micronaut: 'CM-1' },

      'S2:execute':     { nextState: 'S3', action: 'compute',  micronaut: 'MM-1' },
      'S2:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },
      'S2:reset':       { nextState: 'S0', action: 'pop',      micronaut: 'CM-1' },

      'S3:complete':    { nextState: 'S5', action: 'return',   micronaut: 'SM-1' },
      'S3:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },
      'S3:reset':       { nextState: 'S0', action: 'pop',      micronaut: 'CM-1' },

      'S4:reset':       { nextState: 'S0', action: 'recover',  micronaut: 'VM-2' },
      'S4:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },

      'S5:reset':       { nextState: 'S0', action: 'idle',     micronaut: 'CM-1' },
      'S5:error':       { nextState: 'S4', action: 'trap',     micronaut: 'XM-1' },
    };
  }

  /**
   * Look up a transition deterministically
   * @param {string} currentState - Current state ID (S0–S5)
   * @param {string} inputSymbol - Input symbol (parse, transform, execute, error, etc.)
   * @returns {object} Transition rule {nextState, action, micronaut}
   */
  lookup(currentState, inputSymbol) {
    const key = `${currentState}:${inputSymbol}`;
    const transition = this.transitions[key];

    if (!transition) {
      throw new Error(
        `Undefined transition: δ(${currentState}, ${inputSymbol}). ` +
        `This violates the DFA definition.`
      );
    }

    return transition;
  }

  /**
   * Get all valid inputs for a given state
   */
  getValidInputs(currentState) {
    return Object.keys(this.transitions)
      .filter(key => key.startsWith(currentState + ':'))
      .map(key => key.split(':')[1]);
  }
}

/**
 * Input Tape — Read-only sequence of symbols
 * Mimics a Turing machine tape; pointer advances left-to-right
 */
class InputTape {
  constructor(symbols = []) {
    this.symbols = symbols;  // Array of input symbols
    this.pointer = 0;        // Current read position
  }

  /**
   * Read current symbol without advancing
   */
  peek() {
    return this.pointer < this.symbols.length
      ? this.symbols[this.pointer]
      : '$'; // End-of-tape marker
  }

  /**
   * Read current symbol and advance pointer
   */
  read() {
    const symbol = this.peek();
    if (this.pointer < this.symbols.length) {
      this.pointer += 1;
    }
    return symbol;
  }

  /**
   * Reset tape to beginning
   */
  reset() {
    this.pointer = 0;
  }

  /**
   * Get full tape contents (for hashing, proofs)
   */
  contents() {
    return this.symbols.join('');
  }

  /**
   * Deterministic hash of tape contents
   */
  hash() {
    return crypto
      .createHash('sha256')
      .update(this.contents())
      .digest('hex');
  }
}

/**
 * Stack — LIFO data structure for nested scope/context
 */
class ComputeStack {
  constructor(maxDepth = 256) {
    this.frames = [];
    this.pointer = -1;
    this.maxDepth = maxDepth;
  }

  /**
   * Push a scope frame onto the stack
   */
  push(frame) {
    if (this.frames.length >= this.maxDepth) {
      throw new Error(`Stack overflow: exceeded max depth ${this.maxDepth}`);
    }
    this.frames.push(frame);
    this.pointer += 1;
  }

  /**
   * Pop the top frame from the stack
   */
  pop() {
    if (this.pointer < 0) {
      throw new Error('Stack underflow: cannot pop from empty stack');
    }
    const frame = this.frames[this.pointer];
    this.pointer -= 1;
    return frame;
  }

  /**
   * Peek at the top frame without popping
   */
  peek() {
    return this.pointer >= 0 ? this.frames[this.pointer] : null;
  }

  /**
   * Get current stack depth
   */
  depth() {
    return this.pointer + 1;
  }

  /**
   * Deterministic hash of stack contents
   */
  hash() {
    const contents = JSON.stringify(this.frames);
    return crypto
      .createHash('sha256')
      .update(contents)
      .digest('hex');
  }

  /**
   * Clear the stack
   */
  clear() {
    this.frames = [];
    this.pointer = -1;
  }
}

/**
 * Register File — Fixed 4-register architecture
 * ACC (accumulator), PC (program counter), IR (instruction register), FLAGS
 */
class RegisterFile {
  constructor() {
    this.registers = {
      ACC:   0,      // Accumulator (working value)
      PC:    0,      // Program counter (instruction pointer)
      IR:    '',     // Instruction register (current instruction string)
      FLAGS: 0b0000, // Flags: Z(zero), C(carry), N(negative), O(overflow)
    };
  }

  /**
   * Read a register value
   */
  read(name) {
    if (!(name in this.registers)) {
      throw new Error(`Unknown register: ${name}`);
    }
    return this.registers[name];
  }

  /**
   * Write a register value
   */
  write(name, value) {
    if (!(name in this.registers)) {
      throw new Error(`Unknown register: ${name}`);
    }
    this.registers[name] = value;

    // Auto-update FLAGS based on ACC
    if (name === 'ACC') {
      const zeroFlag = value === 0 ? 1 : 0;
      const signFlag = value < 0 ? 1 : 0;
      this.registers.FLAGS = (this.registers.FLAGS & 0b1100) | (zeroFlag << 1) | signFlag;
    }
  }

  /**
   * Increment program counter
   */
  nextInstruction() {
    this.registers.PC += 1;
  }

  /**
   * Check a flag bit
   */
  checkFlag(flagName) {
    const flagBits = { Z: 2, C: 4, N: 1, O: 8 };
    if (!(flagName in flagBits)) {
      throw new Error(`Unknown flag: ${flagName}`);
    }
    return (this.registers.FLAGS & flagBits[flagName]) !== 0;
  }

  /**
   * Deterministic hash of register state
   */
  hash() {
    const contents = JSON.stringify(this.registers);
    return crypto
      .createHash('sha256')
      .update(contents)
      .digest('hex');
  }

  /**
   * Reset all registers
   */
  reset() {
    this.registers = {
      ACC:   0,
      PC:    0,
      IR:    '',
      FLAGS: 0b0000,
    };
  }
}

/**
 * ClusterNodeDFA — Single Cluster Node with Embedded DFA
 *
 * A cluster node is:
 *   - An addressable location in the 10x10x10 grid (nodeId)
 *   - A fold-scoped compute unit
 *   - An autonomous DFA executor
 *
 * No external orchestration; the node processes its input tape deterministically.
 */
class ClusterNodeDFA {
  constructor(nodeId, fold = '⟁COMPUTE_FOLD⟁', type = 'compute') {
    this.nodeId = nodeId;
    this.fold = fold;
    this.type = type;
    this.color = '#00ff88';

    // DFA components
    this.stateDefinitions = {
      S0: new DFAState('S0', 'IDLE', 'initial', '#16f2aa'),
      S1: new DFAState('S1', 'PARSE', 'normal', '#ffaa00'),
      S2: new DFAState('S2', 'TRANSFORM', 'normal', '#9966ff'),
      S3: new DFAState('S3', 'EXECUTE', 'normal', '#00e0ff'),
      S4: new DFAState('S4', 'ERROR', 'error', '#ff6b6b'),
      S5: new DFAState('S5', 'COMPLETE', 'accepting', '#4ecdc4'),
    };

    this.currentState = 'S0';
    this.transitionFunction = new TransitionFunction();
    this.inputTape = null;
    this.stack = new ComputeStack();
    this.registers = new RegisterFile();
    this.controlGateOpen = false;

    // Execution trace for proof
    this.executionTrace = [];
    this.stateHistory = [];
  }

  /**
   * Check if control gate is open (V2 verifier rule)
   */
  requireControlGate() {
    if (!this.controlGateOpen) {
      throw new Error(
        'V2 violation: control gate must be open before state transition. ' +
        'Execute openControlGate() first.'
      );
    }
  }

  /**
   * Open control gate with explicit authorize record (V2 compliance)
   */
  openControlGate(authorizeRecord = {}) {
    if (!authorizeRecord.timestamp) {
      authorizeRecord.timestamp = Date.now();
    }
    this.controlGateOpen = true;
    this.executionTrace.push({
      event: 'gate_open',
      nodeId: this.nodeId,
      authorizeRecord,
    });
  }

  /**
   * Close control gate (defensive: prevents unauthorized state changes)
   */
  closeControlGate() {
    this.controlGateOpen = false;
  }

  /**
   * Set the input tape for this node
   */
  setInputTape(symbols) {
    this.inputTape = new InputTape(symbols);
  }

  /**
   * Execute a single transition step
   * Reads next input symbol from tape and applies δ
   */
  step() {
    // Read next input symbol
    if (!this.inputTape) {
      throw new Error('No input tape set. Call setInputTape() first.');
    }

    const inputSymbol = this.inputTape.peek();

    // Look up transition
    let transition;
    try {
      transition = this.transitionFunction.lookup(this.currentState, inputSymbol);
    } catch (e) {
      // Undefined transition → ERROR state
      transition = this.transitionFunction.lookup(this.currentState, 'error');
    }

    // Check V2 gate for state changes
    if (transition.action !== 'idle') {
      this.requireControlGate();
    }

    // Record state exit
    this.stateDefinitions[this.currentState].exit();

    // Execute action (side effects)
    this._executeAction(transition.action, inputSymbol, transition.micronaut);

    // Update state
    this.currentState = transition.nextState;

    // Record state entry
    this.stateDefinitions[this.currentState].enter();

    // Record in trace
    this.stateHistory.push({
      from: this.currentState,
      to: transition.nextState,
      input: inputSymbol,
      action: transition.action,
      micronaut: transition.micronaut,
      timestamp: Date.now(),
    });

    // Advance tape pointer
    this.inputTape.read();

    return {
      prevState: this.currentState,
      nextState: transition.nextState,
      inputSymbol,
      action: transition.action,
      micronaut: transition.micronaut,
    };
  }

  /**
   * Execute the action associated with a transition
   */
  _executeAction(action, inputSymbol, micronaut) {
    switch (action) {
      case 'push':
        // PARSE: push a scope frame onto the stack
        this.stack.push({
          action: 'parse',
          input: inputSymbol,
          depth: this.stack.depth() + 1,
        });
        this.executionTrace.push({
          event: 'push',
          nodeId: this.nodeId,
          stackDepth: this.stack.depth(),
          micronaut,
        });
        break;

      case 'pop':
        // Reset or complete: pop a scope frame
        try {
          this.stack.pop();
        } catch (e) {
          // Stack underflow → ERROR
          this.currentState = 'S4';
        }
        this.executionTrace.push({
          event: 'pop',
          nodeId: this.nodeId,
          stackDepth: this.stack.depth(),
          micronaut,
        });
        break;

      case 'rewrite':
        // TRANSFORM: mutate accumulator
        const frame = this.stack.peek();
        if (frame) {
          frame.transform = inputSymbol;
        }
        this.registers.write('ACC', this.registers.read('ACC') + 1);
        this.executionTrace.push({
          event: 'rewrite',
          nodeId: this.nodeId,
          accumulator: this.registers.read('ACC'),
          micronaut,
        });
        break;

      case 'compute':
        // EXECUTE: perform computation (accumulator manipulation)
        const result = Math.abs(this.registers.read('ACC')) * 2;
        this.registers.write('ACC', result);
        this.registers.nextInstruction();
        this.executionTrace.push({
          event: 'compute',
          nodeId: this.nodeId,
          accumulator: this.registers.read('ACC'),
          pc: this.registers.read('PC'),
          micronaut,
        });
        break;

      case 'return':
        // COMPLETE: prepare return value
        this.registers.nextInstruction();
        this.executionTrace.push({
          event: 'return',
          nodeId: this.nodeId,
          returnValue: this.registers.read('ACC'),
          micronaut,
        });
        break;

      case 'trap':
        // ERROR: record error and stay in error state
        this.executionTrace.push({
          event: 'trap',
          nodeId: this.nodeId,
          errorInput: inputSymbol,
          currentState: this.currentState,
          micronaut,
        });
        break;

      case 'recover':
        // Reset from ERROR: clear stack and registers
        this.stack.clear();
        this.registers.reset();
        this.inputTape.reset();
        this.executionTrace.push({
          event: 'recover',
          nodeId: this.nodeId,
          micronaut,
        });
        break;

      case 'idle':
        // No-op transition
        this.executionTrace.push({
          event: 'idle',
          nodeId: this.nodeId,
          micronaut,
        });
        break;

      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  /**
   * Run the DFA to completion or until halt
   * Executes all steps until S5 (complete) or S4 (error) or end of tape
   */
  run() {
    if (!this.inputTape) {
      throw new Error('No input tape set. Call setInputTape() first.');
    }

    this.openControlGate({ reason: 'run_start' });

    while (
      this.currentState !== 'S5' &&
      this.currentState !== 'S4' &&
      this.inputTape.pointer < this.inputTape.symbols.length
    ) {
      this.step();
    }

    this.closeControlGate();

    return {
      nodeId: this.nodeId,
      finalState: this.currentState,
      accumulator: this.registers.read('ACC'),
      stackDepth: this.stack.depth(),
      pc: this.registers.read('PC'),
      stateHistory: this.stateHistory,
      executionTrace: this.executionTrace,
      tapeHash: this.inputTape.hash(),
      stackHash: this.stack.hash(),
      registerHash: this.registers.hash(),
      deterministic: true,
    };
  }

  /**
   * Get overall execution hash (V6 compliance)
   * Same input → same node → same hash
   * Excludes timestamps for true determinism
   */
  executionHash() {
    // Strip timestamps from stateHistory for deterministic hashing
    const deterministicHistory = this.stateHistory.map(s => ({
      from: s.from,
      to: s.to,
      input: s.input,
      action: s.action,
      micronaut: s.micronaut,
    }));

    const hashInput = JSON.stringify({
      tape: this.inputTape?.hash(),
      stack: this.stack.hash(),
      registers: this.registers.hash(),
      stateHistory: deterministicHistory,
      finalState: this.currentState,
    });
    return crypto
      .createHash('sha256')
      .update(hashInput)
      .digest('hex');
  }

  /**
   * Export state for persistence/inspection
   */
  toJSON() {
    return {
      nodeId: this.nodeId,
      fold: this.fold,
      type: this.type,
      currentState: this.currentState,
      registers: this.registers.registers,
      stackDepth: this.stack.depth(),
      tapePointer: this.inputTape?.pointer || 0,
      stateHistory: this.stateHistory,
      executionTrace: this.executionTrace,
      hash: this.executionHash(),
    };
  }
}

/**
 * Multi-Node Cluster State Machine Orchestrator
 * Coordinates multiple ClusterNodeDFA instances
 */
class ClusterStateMachineOrchestrator {
  constructor(clusterSize = 1000) {
    this.nodes = new Map();
    this.nodeIndex = 0;
    this.clusterSize = clusterSize;
    this._initializeCluster();
  }

  /**
   * Initialize 1000-node cluster with type-based allocation
   */
  _initializeCluster() {
    const types = [
      { name: 'compute', count: 500, fold: '⟁COMPUTE_FOLD⟁' },
      { name: 'routing', count: 200, fold: '⟁DATA_FOLD⟁' },
      { name: 'storage', count: 150, fold: '⟁STORAGE_FOLD⟁' },
      { name: 'verification', count: 100, fold: '⟁META_FOLD⟁' },
      { name: 'control', count: 50, fold: '⟁CONTROL_FOLD⟁' },
    ];

    for (const { name, count, fold } of types) {
      for (let i = 0; i < count; i++) {
        const nodeId = `node_${this.nodeIndex}`;
        const node = new ClusterNodeDFA(nodeId, fold, name);
        this.nodes.set(nodeId, node);
        this.nodeIndex += 1;
      }
    }
  }

  /**
   * Get a node by ID
   */
  getNode(nodeId) {
    return this.nodes.get(nodeId);
  }

  /**
   * Get all nodes of a specific type
   */
  getNodesByType(type) {
    return Array.from(this.nodes.values()).filter(node => node.type === type);
  }

  /**
   * Get all nodes in a specific fold
   */
  getNodesByFold(fold) {
    return Array.from(this.nodes.values()).filter(node => node.fold === fold);
  }

  /**
   * Execute a computation on a specific node
   */
  executeOnNode(nodeId, inputSymbols) {
    const node = this.getNode(nodeId);
    if (!node) {
      throw new Error(`Node not found: ${nodeId}`);
    }

    node.setInputTape(inputSymbols);
    return node.run();
  }

  /**
   * Execute on first available node of a specific type
   */
  executeOnNodeType(type, inputSymbols) {
    const nodes = this.getNodesByType(type);
    if (nodes.length === 0) {
      throw new Error(`No nodes of type: ${type}`);
    }

    // Deterministic: always use first available node
    const node = nodes[0];
    return this.executeOnNode(node.nodeId, inputSymbols);
  }

  /**
   * Broadcast execution across all nodes in a fold
   */
  broadcastOnFold(fold, inputSymbols) {
    const nodes = this.getNodesByFold(fold);
    const results = [];

    for (const node of nodes) {
      try {
        const result = this.executeOnNode(node.nodeId, inputSymbols);
        results.push(result);
      } catch (e) {
        results.push({
          nodeId: node.nodeId,
          error: e.message,
        });
      }
    }

    return {
      fold,
      nodeCount: nodes.length,
      results,
      successCount: results.filter(r => !r.error).length,
      failureCount: results.filter(r => r.error).length,
    };
  }

  /**
   * Get cluster statistics
   */
  clusterStats() {
    const types = {};
    const folds = {};

    for (const node of this.nodes.values()) {
      types[node.type] = (types[node.type] || 0) + 1;
      folds[node.fold] = (folds[node.fold] || 0) + 1;
    }

    return {
      totalNodes: this.nodes.size,
      byType: types,
      byFold: folds,
    };
  }

  /**
   * Export full cluster state
   */
  clusterSnapshot() {
    const snapshot = {
      totalNodes: this.nodes.size,
      timestamp: Date.now(),
      stats: this.clusterStats(),
      nodes: Array.from(this.nodes.values()).map(node => node.toJSON()),
    };
    return snapshot;
  }
}

// Export for use in other modules
export { ClusterNodeDFA, ClusterStateMachineOrchestrator, TransitionFunction, InputTape, ComputeStack, RegisterFile, DFAState };
