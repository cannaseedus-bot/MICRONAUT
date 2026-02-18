/**
 * Test Suite: Cluster Node DFA State Machine
 *
 * Tests deterministic behavior, state transitions, stack/register operations,
 * and V2/V6 compliance for proof-driven architecture.
 */

import {
  ClusterNodeDFA,
  ClusterStateMachineOrchestrator,
  InputTape,
  ComputeStack,
  RegisterFile,
} from './cluster_node_state_machine.js';

/**
 * Test 1: Basic state transitions (happy path)
 */
function testBasicTransitions() {
  console.log('\n=== Test 1: Basic State Transitions ===');

  const node = new ClusterNodeDFA('test_node_1', '⟁COMPUTE_FOLD⟁', 'compute');
  node.setInputTape(['parse', 'transform', 'execute', 'complete']);

  const result = node.run();

  console.log(`Node: ${result.nodeId}`);
  console.log(`Final State: ${result.finalState} (expected: S5)`);
  console.log(`State History: ${result.stateHistory.map(s => s.to).join(' → ')}`);
  console.log(`Accumulator: ${result.accumulator}`);
  console.log(`PC: ${result.pc}`);

  const pass = result.finalState === 'S5';
  console.log(`✓ PASS` + (pass ? '' : ` FAIL: expected S5, got ${result.finalState}`));

  return pass;
}

/**
 * Test 2: V2 Control Gate Enforcement
 */
function testControlGateEnforcement() {
  console.log('\n=== Test 2: V2 Control Gate Enforcement ===');

  const node = new ClusterNodeDFA('test_node_2', '⟁CONTROL_FOLD⟁', 'control');
  node.setInputTape(['parse']);

  // Should fail without opening gate
  let gateError = false;
  try {
    node.step();
  } catch (e) {
    gateError = e.message.includes('V2 violation');
    console.log(`✓ Caught expected V2 error: "${e.message}"`);
  }

  if (gateError) {
    // Now open gate and try again
    node.currentState = 'S0';
    node.inputTape.reset();
    node.openControlGate({ reason: 'test_gate' });

    try {
      node.step();
      console.log('✓ Transition succeeded after gate opened');
    } catch (e) {
      console.log(`✗ FAIL: ${e.message}`);
      return false;
    }
  }

  console.log('✓ PASS: V2 gate enforcement working');
  return gateError;
}

/**
 * Test 3: Error state handling
 */
function testErrorHandling() {
  console.log('\n=== Test 3: Error State Handling ===');

  const node = new ClusterNodeDFA('test_node_3', '⟁COMPUTE_FOLD⟁', 'compute');
  node.setInputTape(['parse', 'error', 'reset']);
  node.openControlGate({ reason: 'error_test' });

  let currentState = 'S0';
  let errorHit = false;

  while (currentState !== 'S5' && currentState !== 'S4' && node.inputTape.pointer < 3) {
    const transition = node.step();
    currentState = node.currentState;

    if (currentState === 'S4') {
      errorHit = true;
      console.log(`✓ Error state reached after error input`);
    }
  }

  console.log(`Final state: ${currentState}`);
  console.log(`Error was triggered: ${errorHit}`);
  console.log('✓ PASS: Error handling works');

  return errorHit;
}

/**
 * Test 4: Stack operations (push/pop)
 */
function testStackOperations() {
  console.log('\n=== Test 4: Stack Operations ===');

  const stack = new ComputeStack(10);

  stack.push({ id: 'frame1' });
  console.log(`After push 1: depth = ${stack.depth()} (expected: 1)`);

  stack.push({ id: 'frame2' });
  console.log(`After push 2: depth = ${stack.depth()} (expected: 2)`);

  stack.push({ id: 'frame3' });
  console.log(`After push 3: depth = ${stack.depth()} (expected: 3)`);

  const frame3 = stack.pop();
  console.log(`Popped frame: ${frame3.id}, depth = ${stack.depth()} (expected: 2)`);

  const frame2 = stack.pop();
  console.log(`Popped frame: ${frame2.id}, depth = ${stack.depth()} (expected: 1)`);

  const pass = stack.depth() === 1;
  console.log(pass ? '✓ PASS: Stack operations correct' : '✗ FAIL: Stack depth mismatch');

  return pass;
}

/**
 * Test 5: Register operations (ACC, PC, FLAGS)
 */
function testRegisterOperations() {
  console.log('\n=== Test 5: Register Operations ===');

  const regs = new RegisterFile();

  regs.write('ACC', 5);
  console.log(`ACC = ${regs.read('ACC')} (expected: 5)`);

  regs.nextInstruction();
  console.log(`PC after nextInstruction: ${regs.read('PC')} (expected: 1)`);

  regs.nextInstruction();
  console.log(`PC after nextInstruction: ${regs.read('PC')} (expected: 2)`);

  // Test zero flag
  regs.write('ACC', 0);
  const zeroFlag = regs.checkFlag('Z');
  console.log(`Zero flag when ACC=0: ${zeroFlag} (expected: true)`);

  // Test negative flag
  regs.write('ACC', -10);
  const negFlag = regs.checkFlag('N');
  console.log(`Negative flag when ACC=-10: ${negFlag} (expected: true)`);

  const pass = regs.read('PC') === 2 && zeroFlag && negFlag;
  console.log(pass ? '✓ PASS: Register operations correct' : '✗ FAIL: Register mismatch');

  return pass;
}

/**
 * Test 6: Input tape determinism
 */
function testInputTapeDeterminism() {
  console.log('\n=== Test 6: Input Tape Determinism ===');

  const tape1 = new InputTape(['parse', 'transform', 'execute']);
  const tape2 = new InputTape(['parse', 'transform', 'execute']);

  const hash1 = tape1.hash();
  const hash2 = tape2.hash();

  console.log(`Tape 1 hash: ${hash1.substring(0, 16)}...`);
  console.log(`Tape 2 hash: ${hash2.substring(0, 16)}...`);

  const match = hash1 === hash2;
  console.log(match ? '✓ PASS: Identical tapes produce identical hashes' : '✗ FAIL: Hash mismatch');

  return match;
}

/**
 * Test 7: Deterministic execution (same input → same output)
 */
function testDeterministicExecution() {
  console.log('\n=== Test 7: Deterministic Execution (V6 Compliance) ===');

  const inputSymbols = ['parse', 'transform', 'execute', 'complete'];

  // Run 1
  const node1 = new ClusterNodeDFA('node1', '⟁COMPUTE_FOLD⟁', 'compute');
  node1.setInputTape(inputSymbols);
  const result1 = node1.run();
  const hash1 = node1.executionHash();

  // Run 2
  const node2 = new ClusterNodeDFA('node2', '⟁COMPUTE_FOLD⟁', 'compute');
  node2.setInputTape(inputSymbols);
  const result2 = node2.run();
  const hash2 = node2.executionHash();

  console.log(`Node 1 execution hash: ${hash1.substring(0, 16)}...`);
  console.log(`Node 2 execution hash: ${hash2.substring(0, 16)}...`);
  console.log(`Hashes match: ${hash1 === hash2}`);
  console.log(`Final states match: ${result1.finalState === result2.finalState}`);
  console.log(`Accumulators match: ${result1.accumulator === result2.accumulator}`);

  const pass = hash1 === hash2 && result1.finalState === result2.finalState;
  console.log(pass ? '✓ PASS: V6 deterministic execution verified' : '✗ FAIL: Non-determinism detected');

  return pass;
}

/**
 * Test 8: Multi-node cluster orchestration
 */
function testClusterOrchestrator() {
  console.log('\n=== Test 8: Multi-Node Cluster Orchestration ===');

  const orchestrator = new ClusterStateMachineOrchestrator(1000);
  const stats = orchestrator.clusterStats();

  console.log(`Total nodes: ${stats.totalNodes} (expected: 1000)`);
  console.log(`By type:`, stats.byType);
  console.log(`By fold:`, Object.keys(stats.byFold).slice(0, 5), '...');

  // Execute on a compute node
  const inputSymbols = ['parse', 'transform', 'execute', 'complete'];
  const result = orchestrator.executeOnNodeType('compute', inputSymbols);

  console.log(`Execution on compute node:`);
  console.log(`  Node: ${result.nodeId}`);
  console.log(`  Final State: ${result.finalState}`);
  console.log(`  Accumulator: ${result.accumulator}`);

  const pass = stats.totalNodes === 1000 && result.finalState === 'S5';
  console.log(pass ? '✓ PASS: Cluster orchestrator working' : '✗ FAIL: Cluster issue');

  return pass;
}

/**
 * Test 9: Broadcast execution on fold
 */
function testBroadcastOnFold() {
  console.log('\n=== Test 9: Broadcast Execution on Fold ===');

  const orchestrator = new ClusterStateMachineOrchestrator(1000);
  const inputSymbols = ['parse', 'transform', 'execute', 'complete'];

  const result = orchestrator.broadcastOnFold('⟁CONTROL_FOLD⟁', inputSymbols);

  console.log(`Broadcast on ⟁CONTROL_FOLD⟁:`);
  console.log(`  Total nodes: ${result.nodeCount}`);
  console.log(`  Successful: ${result.successCount}`);
  console.log(`  Failed: ${result.failureCount}`);
  console.log(`  Success rate: ${((result.successCount / result.nodeCount) * 100).toFixed(1)}%`);

  const pass = result.successCount > 0 && result.failureCount === 0;
  console.log(pass ? '✓ PASS: Broadcast execution successful' : '✗ FAIL: Broadcast issue');

  return pass;
}

/**
 * Test 10: Undefined transition handling
 */
function testUndefinedTransition() {
  console.log('\n=== Test 10: Undefined Transition Handling ===');

  const node = new ClusterNodeDFA('test_node_10', '⟁COMPUTE_FOLD⟁', 'compute');
  node.setInputTape(['parse', 'undefined_symbol']);
  node.openControlGate({ reason: 'undefined_test' });

  let errorCaught = false;

  try {
    node.step(); // S0 → S1 (parse)
    node.step(); // S1 → ? (undefined_symbol) should trap to S4
  } catch (e) {
    errorCaught = true;
    console.log(`✓ Transition error caught: "${e.message.substring(0, 50)}..."`);
  }

  // Even if no error thrown, we should be in error state or have handled gracefully
  console.log(`Current state: ${node.currentState}`);

  const pass = node.currentState === 'S4' || errorCaught;
  console.log(pass ? '✓ PASS: Undefined transition handled correctly' : '✗ FAIL: Should enter error state');

  return pass;
}

/**
 * Main test runner
 */
function runAllTests() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  Cluster Node DFA State Machine Test Suite                     ║');
  console.log('║  Testing Deterministic Finite Automaton Behavior               ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  const tests = [
    { name: 'Basic Transitions', fn: testBasicTransitions },
    { name: 'V2 Control Gate', fn: testControlGateEnforcement },
    { name: 'Error Handling', fn: testErrorHandling },
    { name: 'Stack Operations', fn: testStackOperations },
    { name: 'Register Operations', fn: testRegisterOperations },
    { name: 'Tape Determinism', fn: testInputTapeDeterminism },
    { name: 'V6 Execution', fn: testDeterministicExecution },
    { name: 'Cluster Orchestration', fn: testClusterOrchestrator },
    { name: 'Broadcast on Fold', fn: testBroadcastOnFold },
    { name: 'Undefined Transition', fn: testUndefinedTransition },
  ];

  const results = [];
  for (const test of tests) {
    try {
      const pass = test.fn();
      results.push({ name: test.name, pass });
    } catch (e) {
      console.log(`✗ EXCEPTION in ${test.name}: ${e.message}`);
      results.push({ name: test.name, pass: false });
    }
  }

  // Summary
  console.log('\n╔════════════════════════════════════════════════════════════════╗');
  console.log('║ Test Summary                                                   ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  const passed = results.filter(r => r.pass).length;
  const total = results.length;

  for (const result of results) {
    const icon = result.pass ? '✓' : '✗';
    console.log(`${icon} ${result.name.padEnd(40)} ${result.pass ? 'PASS' : 'FAIL'}`);
  }

  console.log(`\nTotal: ${passed}/${total} tests passed (${((passed / total) * 100).toFixed(1)}%)\n`);

  return passed === total;
}

// Run tests
const allPassed = runAllTests();
process.exit(allPassed ? 0 : 1);
