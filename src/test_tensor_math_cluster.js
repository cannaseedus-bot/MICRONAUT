/**
 * Test Suite: Tensor Math Cluster Integration
 *
 * Validates pi-geometry tensors, neural weight matrices, inference operations,
 * cognitive graph propagation, matrix operations, inference calculus,
 * quantum states, and full cluster integration.
 */

import {
  PI, TAU, PHI,
  PiGeometryTensor,
  TensorValue,
  NeuralWeightMatrix,
  InferenceOperations,
  CognitiveNode,
  CognitiveGraph,
  MatrixOperations,
  InferenceCalculus,
  SV3GTensorMapping,
  QuantumCognitiveState,
  ValidationMetrics,
  TensorClusterIntegration,
} from './tensor_math_cluster.js';

import { InferenceCluster, createInferenceCluster } from './inference_cluster.js';

// -------------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------------

function approxEqual(a, b, epsilon = 1e-6) {
  return Math.abs(a - b) < epsilon;
}

// -------------------------------------------------------------------------
// Test 1: Pi-Geometry Tensor (4D manifold)
// -------------------------------------------------------------------------

function testPiGeometryTensor() {
  console.log('\n=== Test 1: Pi-Geometry Tensor (4D Manifold) ===');

  const tensor = new PiGeometryTensor();

  // Verify rank and dimensions
  console.log(`Rank: ${tensor.rank} (expected: 4)`);
  console.log(`Dimensions: ${tensor.dimensions.length} (expected: 4)`);

  // Verify dimension types
  const types = tensor.dimensions.map(d => d.type);
  console.log(`Types: ${types.join(', ')}`);

  // Spatial sampling
  const w0 = tensor.sampleSpatial(0);
  const w1 = tensor.sampleSpatial(PI / 4);
  const w2 = tensor.sampleSpatial(PI / 2);
  console.log(`Spatial weights: θ=0 → ${w0.toFixed(4)}, θ=π/4 → ${w1.toFixed(4)}, θ=π/2 → ${w2.toFixed(4)}`);

  // Temporal harmonic
  const h = tensor.temporalHarmonic(0.5);
  console.log(`Temporal harmonic at τ=0.5: ${h.toFixed(4)}`);

  // Semantic convergence
  const s1 = tensor.semanticConvergence(1);
  const s5 = tensor.semanticConvergence(5);
  console.log(`Semantic convergence: depth=1 → ${s1.toFixed(4)}, depth=5 → ${s5.toFixed(6)}`);

  // 4D evaluation
  const val = tensor.evaluate(PI / 4, 0.5, 2, 0.8);
  console.log(`4D evaluation: ${val.toFixed(6)}`);

  // Deterministic hash
  const hash = tensor.hash();
  console.log(`Hash: ${hash.substring(0, 16)}...`);

  const pass = tensor.rank === 4 &&
               tensor.dimensions.length === 4 &&
               w0 === PI / 2 &&
               typeof val === 'number' &&
               !isNaN(val);

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 2: Neural Weight Matrix
// -------------------------------------------------------------------------

function testNeuralWeightMatrix() {
  console.log('\n=== Test 2: Neural Weight Matrix (Pi-Spherical) ===');

  const matrix = new NeuralWeightMatrix();

  console.log(`Layers: ${matrix.layers.length} (expected: 2)`);
  console.log(`Layer types: ${matrix.layers.map(l => l.type).join(', ')}`);

  // Attention weights
  const attWeights = matrix.getAttentionWeights();
  console.log(`Attention weights: [${attWeights.map(w => w.toFixed(4)).join(', ')}]`);

  // Torus sampling
  const torusVal = matrix.sampleTorus(0, 0);
  console.log(`Torus sample at (0,0): ${torusVal.toFixed(4)}`);

  // Individual weight access
  const w00 = matrix.getWeight(0, 0, 0);
  const w01 = matrix.getWeight(0, 0, 1);
  console.log(`Weight[0][0,0]: ${w00.toFixed(4)}, Weight[0][0,1]: ${w01.toFixed(4)}`);

  // Parameter count
  const params = matrix.parameterCount();
  console.log(`Parameter count: ${params}`);

  // Hash
  const hash = matrix.hash();
  console.log(`Hash: ${hash.substring(0, 16)}...`);

  const pass = matrix.layers.length === 2 &&
               attWeights.length === 2 &&
               params > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 3: TensorValue (spherical coordinates)
// -------------------------------------------------------------------------

function testTensorValue() {
  console.log('\n=== Test 3: TensorValue (Spherical Coordinates) ===');

  const tv = new TensorValue(1.0, PI / 6, PI / 4, 0.75, PI / 3, PI / 8);

  // Cartesian conversion
  const cart = tv.toCartesian();
  console.log(`Cartesian: (${cart.x.toFixed(4)}, ${cart.y.toFixed(4)}, ${cart.z.toFixed(4)})`);

  // Effective weight
  const ew = tv.effectiveWeight();
  console.log(`Effective weight: ${ew.toFixed(4)}`);

  // Verify: magnitude * cos(phase) * (1 - entropy/π)
  const expected = 0.75 * Math.cos(PI / 3) * (1 - (PI / 8) / PI);
  console.log(`Expected: ${expected.toFixed(4)}`);

  const pass = approxEqual(ew, expected);
  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 4: Inference Operations
// -------------------------------------------------------------------------

function testInferenceOperations() {
  console.log('\n=== Test 4: Inference Operations ===');

  const ops = new InferenceOperations();
  console.log(`Registered operations: ${ops.listOperations().join(', ')}`);

  // Attention-pi-transform
  const Q = [0.5, 0.3, 0.7];
  const K = [0.4, 0.6, 0.2];
  const V = [1.0, 0.8, 0.6];
  const attResult = ops.execute('attention-pi-transform', Q, K, V, 3);
  console.log(`Attention output: [${attResult.output.map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`Attention weights: [${attResult.attention.map(v => v.toFixed(4)).join(', ')}]`);

  // Tensor contraction
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  const contracted = ops.execute('tensor-contraction-pi', A, B, 1);
  console.log(`Tensor contraction: [${contracted.map(v => v.toFixed(4)).join(', ')}]`);

  // Weight update
  const weights = [0.5, 0.3, 0.7];
  const gradients = [0.1, -0.2, 0.05];
  const updated = ops.execute('weight-update-pi', weights, gradients);
  console.log(`Updated weights: [${updated.map(v => v.toFixed(6)).join(', ')}]`);

  const pass = attResult.output.length === 3 &&
               contracted.length === 2 &&
               updated.length === 3;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 5: Cognitive Graph
// -------------------------------------------------------------------------

function testCognitiveGraph() {
  console.log('\n=== Test 5: Cognitive Reasoning Graph ===');

  const graph = new CognitiveGraph();

  graph.addNode('c1', 1.0, PI / 3, PI / 6, 'sigmoid-pi');
  graph.addNode('c2', 0.8, 2 * PI / 3, PI / 4, 'tanh-pi');
  graph.addNode('c3', 0.6, PI / 2, PI / 3, 'sigmoid-pi');

  graph.addEdge('c1', 'c2', PI / 3, 2 * PI / 5, 3 * PI / 4);
  graph.addEdge('c2', 'c3', PI / 4, PI / 3, PI / 2);

  console.log(`Nodes: ${graph.nodes.size}, Edges: ${graph.edges.length}`);

  // Run inference
  const inputs = new Map([['c1', 0.8], ['c2', 0.3], ['c3', 0.1]]);
  const result = graph.infer(inputs, 3);

  console.log(`Inference steps: ${result.trace.length}`);
  for (const [nodeId, activation] of Object.entries(result.finalActivations)) {
    console.log(`  ${nodeId}: ${activation.toFixed(4)}`);
  }

  const pass = result.trace.length === 3 &&
               Object.keys(result.finalActivations).length === 3;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 6: Matrix Operations (Pi-scaled)
// -------------------------------------------------------------------------

function testMatrixOperations() {
  console.log('\n=== Test 6: Matrix Operations (Pi-Scaled) ===');

  // Tensor product
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  const product = MatrixOperations.tensorProductPi(A, B);
  console.log(`Tensor product (2x2):`);
  for (const row of product) {
    console.log(`  [${row.map(v => v.toFixed(4)).join(', ')}]`);
  }

  // Spectral analysis
  const sym = [[2, 1], [1, 3]];
  const spectral = MatrixOperations.spectralAnalysisPi(sym);
  console.log(`Dominant eigenvalue: ${spectral.eigenvalue.toFixed(4)} (bounded by ±π)`);
  console.log(`Eigenvector: [${spectral.eigenvector.map(v => v.toFixed(4)).join(', ')}]`);

  // Determinant
  const det = MatrixOperations.determinant([[1, 2], [3, 4]]);
  console.log(`Determinant of [[1,2],[3,4]]: ${det} (expected: -2)`);

  const pass = product.length === 2 &&
               product[0].length === 2 &&
               Math.abs(spectral.eigenvalue) <= PI &&
               det === -2;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 7: Inference Calculus
// -------------------------------------------------------------------------

function testInferenceCalculus() {
  console.log('\n=== Test 7: Inference Calculus (Bayesian/DS/Fuzzy) ===');

  // Bayesian-pi
  const posterior = InferenceCalculus.bayesianPi(0.6, 0.8, 0.7);
  console.log(`Bayesian-pi posterior: ${posterior.toFixed(4)}`);

  // Dempster-Shafer-pi
  const m1 = { hypothesis: 0.6, universal: 0.4 };
  const m2 = { hypothesis: 0.7, universal: 0.3 };
  const combined = InferenceCalculus.dempsterShaferPi(m1, m2);
  console.log(`DS-pi combined:`, combined ? Object.keys(combined).join(', ') : 'null');
  if (combined && combined.hypothesis) {
    console.log(`  hypothesis belief: ${combined.hypothesis.belief.toFixed(4)}`);
  }

  // Fuzzy-pi
  const muA = [0.8, 0.5, 0.3];
  const muB = [0.6, 0.7, 0.4];
  const rel = [[0.9, 0.3, 0.1], [0.2, 0.8, 0.4], [0.1, 0.3, 0.7]];
  const fuzzy = InferenceCalculus.fuzzyPi(muA, muB, rel);
  console.log(`Fuzzy-pi: degree=${fuzzy.degree.toFixed(4)}, certainty=${fuzzy.certainty.toFixed(4)}`);

  const pass = posterior > 0 &&
               combined !== null &&
               fuzzy.degree > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 8: SV3G Tensor Mapping
// -------------------------------------------------------------------------

function testSV3GMapping() {
  console.log('\n=== Test 8: SV3G Tensor Mapping ===');

  const sv3g = new SV3GTensorMapping();

  // Quantization
  const qx = sv3g.quantize(1.5, 'axisX');
  const qy = sv3g.quantize(3.0, 'axisY');
  console.log(`Quantized X: ${qx.toFixed(4)} (from 1.5)`);
  console.log(`Quantized Y: ${qy.toFixed(4)} (from 3.0)`);

  // Weight node glyph
  const glyph = sv3g.weightNodeGlyph(0.8, 0.7, 0.6, 0.3, 0.5);
  console.log(`Glyph sphere radius: ${glyph.geometry.sphereRadius.toFixed(4)}`);
  console.log(`Glyph fill opacity: ${glyph.attributes.fillOpacity.toFixed(4)}`);

  // Cluster grid mapping
  const gridPos = sv3g.mapToClusterGrid(PI / 3, PI, 0);
  console.log(`Grid position: (${gridPos.x}, ${gridPos.y}, ${gridPos.z})`);

  const pass = typeof qx === 'number' &&
               glyph.geometry.sphereRadius > 0 &&
               gridPos.x >= 0 && gridPos.x < 10;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 9: Quantum Cognitive State
// -------------------------------------------------------------------------

function testQuantumCognitiveState() {
  console.log('\n=== Test 9: Quantum Cognitive State ===');

  const qstate = new QuantumCognitiveState('belief-1', PI / 3, 2 * PI / 3);

  console.log(`P(|0⟩): ${qstate.probability0().toFixed(4)}`);
  console.log(`P(|1⟩): ${qstate.probability1().toFixed(4)}`);
  console.log(`Sum: ${(qstate.probability0() + qstate.probability1()).toFixed(4)} (should be ~1.0)`);

  // Measure
  const measurement = qstate.measure();
  console.log(`Measurement: |${measurement}⟩`);

  // Bloch coordinates
  const bloch = qstate.blochCoordinates();
  console.log(`Bloch: (${bloch.x.toFixed(4)}, ${bloch.y.toFixed(4)}, ${bloch.z.toFixed(4)})`);

  // Apply Hadamard-pi gate
  const before = qstate.probability0();
  qstate.applyHadamardPi();
  console.log(`After Hadamard-pi: P(|0⟩)=${qstate.probability0().toFixed(4)}, P(|1⟩)=${qstate.probability1().toFixed(4)}`);

  // Apply phase shift
  qstate.applyPhaseShiftPi(0.5);
  console.log(`After phase-shift(0.5): P(|0⟩)=${qstate.probability0().toFixed(4)}, P(|1⟩)=${qstate.probability1().toFixed(4)}`);

  const sumProb = qstate.probability0() + qstate.probability1();
  const pass = approxEqual(sumProb, 1.0, 0.01);

  console.log(pass ? '✓ PASS' : `✗ FAIL (probability sum: ${sumProb})`);
  return pass;
}

// -------------------------------------------------------------------------
// Test 10: Validation Metrics
// -------------------------------------------------------------------------

function testValidationMetrics() {
  console.log('\n=== Test 10: Validation Metrics ===');

  // Tensor consistency (pi-scaled values for meaningful consistency)
  const tc = ValidationMetrics.tensorConsistency([PI / 2, PI / 3, PI / 4, PI / 5, PI / 6]);
  console.log(`Tensor consistency: ${tc.toFixed(4)}`);

  // Inference confidence
  const ic = ValidationMetrics.inferenceConfidence(
    (theta) => Math.exp(-theta * theta / 2) / Math.sqrt(TAU)
  );
  console.log(`Inference confidence: ${ic.toFixed(4)}`);

  // Weight stability
  const ws = ValidationMetrics.weightStability([0.5, 0.3, 0.7], [0.49, 0.31, 0.69]);
  console.log(`Weight stability: ${ws.value.toFixed(6)} (acceptable: ${ws.acceptable})`);

  const pass = tc > 0 && tc <= 1 &&
               ic > 0 &&
               ws.acceptable;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 11: TensorClusterIntegration
// -------------------------------------------------------------------------

function testTensorClusterIntegration() {
  console.log('\n=== Test 11: Tensor Cluster Integration ===');

  const integration = new TensorClusterIntegration();

  // Bind some test nodes
  for (let x = 0; x < 3; x++) {
    for (let y = 0; y < 3; y++) {
      for (let z = 0; z < 3; z++) {
        integration.bindNodeTensor(`node_${x}_${y}_${z}`, { x, y, z });
      }
    }
  }

  console.log(`Bound nodes: ${integration.tensorNodeBindings.size} (expected: 27)`);

  // Check a binding
  const b = integration.tensorNodeBindings.get('node_1_1_1');
  console.log(`Node 1,1,1 tensor weight: ${b.tensorWeight.toFixed(6)}`);
  console.log(`  Spatial weight: ${b.spatialWeight.toFixed(4)}`);
  console.log(`  Temporal harmonic: ${b.temporalHarmonic.toFixed(4)}`);
  console.log(`  Semantic depth: ${b.semanticDepth.toFixed(6)}`);

  // Attention computation
  const queryIds = ['node_0_0_0', 'node_1_0_0'];
  const keyIds = ['node_0_1_0', 'node_1_1_0'];
  const valueIds = ['node_0_0_1', 'node_1_0_1', 'node_0_1_1', 'node_1_1_1'];
  const attention = integration.computeAttention(queryIds, keyIds, valueIds);
  console.log(`Attention computed: ${attention.output.length} outputs`);

  // Weight update
  const gradients = [0.01, -0.02, 0.005];
  const updated = integration.updateWeights(['node_0_0_0', 'node_1_1_1', 'node_2_2_2'], gradients);
  console.log(`Weight update: ${updated.length} weights updated`);

  // Cognitive inference
  const cogResult = integration.runCognitiveInference(
    new Map([['concept-1', 0.8], ['concept-2', 0.3]]), 3
  );
  console.log(`Cognitive inference: ${cogResult.trace.length} steps`);

  // Full validation
  const validation = integration.validate();
  console.log(`Validation:`);
  console.log(`  Consistency: ${validation.tensorConsistency.toFixed(4)}`);
  console.log(`  Confidence: ${validation.inferenceConfidence.toFixed(4)}`);
  console.log(`  Stability: acceptable=${validation.weightStability.acceptable}`);
  console.log(`  Valid: ${validation.valid}`);

  const pass = integration.tensorNodeBindings.size === 27 &&
               attention.output.length > 0 &&
               validation.valid;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 12: Full Inference Cluster with Tensor Integration
// -------------------------------------------------------------------------

async function testFullClusterIntegration() {
  console.log('\n=== Test 12: Full Inference Cluster + Tensor Math ===');

  const cluster = createInferenceCluster(10);
  const status = cluster.getStatus();

  console.log(`Total nodes: ${status.totalNodes}`);
  console.log(`Tensor bound: ${status.tensor.boundNodes}`);
  console.log(`Tensor consistency: ${status.tensor.consistency.toFixed(4)}`);
  console.log(`Tensor valid: ${status.tensor.valid}`);

  // Verify all nodes have tensor bindings
  let tensorBound = 0;
  let quantumBound = 0;
  for (const nodeData of cluster.getNodesArray()) {
    if (nodeData.hasTensorBinding) tensorBound++;
    if (nodeData.quantumState) quantumBound++;
  }
  console.log(`Nodes with tensor: ${tensorBound}/${status.totalNodes}`);
  console.log(`Nodes with quantum: ${quantumBound}/${status.totalNodes}`);

  // Run inference with tensor-enhanced pipeline (async)
  const result = await cluster.runInference('compute tensor inference weights for model layer');

  console.log(`Pipeline stages: ${result.stages}`);
  console.log(`Total nodes activated: ${result.totalNodesActivated}`);
  console.log(`Tensor validation: consistency=${result.tensorValidation.consistency.toFixed(4)}, valid=${result.tensorValidation.valid}`);

  // Verify tensor attention was computed in stages
  let tensorStages = 0;
  for (const stage of result.trace) {
    if (stage.tensorAttention && stage.tensorAttention.computed) {
      tensorStages++;
    }
  }
  console.log(`Stages with tensor attention: ${tensorStages}/${result.stages}`);

  // Verify avg tensor weights are populated
  let nonZeroTensorStages = 0;
  for (const stage of result.trace) {
    if (stage.avgTensorWeight !== 0) nonZeroTensorStages++;
  }
  console.log(`Stages with non-zero tensor weight: ${nonZeroTensorStages}/${result.stages}`);

  const pass = tensorBound === status.totalNodes &&
               quantumBound === status.totalNodes &&
               result.tensorValidation.valid &&
               tensorStages > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 13: Deterministic Tensor Hashing (V6)
// -------------------------------------------------------------------------

function testDeterministicTensorHashing() {
  console.log('\n=== Test 13: Deterministic Tensor Hashing (V6) ===');

  // Two independent integrations with same data should produce same hashes
  const int1 = new TensorClusterIntegration();
  const int2 = new TensorClusterIntegration();

  for (let i = 0; i < 5; i++) {
    int1.bindNodeTensor(`node_${i}`, { x: i, y: 0, z: 0 });
    int2.bindNodeTensor(`node_${i}`, { x: i, y: 0, z: 0 });
  }

  const hash1 = int1.hash();
  const hash2 = int2.hash();

  console.log(`Integration 1 hash: ${hash1.substring(0, 16)}...`);
  console.log(`Integration 2 hash: ${hash2.substring(0, 16)}...`);
  console.log(`Hashes match: ${hash1 === hash2}`);

  // Pi tensor hashes
  const pt1 = int1.piTensor.hash();
  const pt2 = int2.piTensor.hash();
  console.log(`PiTensor hashes match: ${pt1 === pt2}`);

  // Weight matrix hashes
  const wm1 = int1.weightMatrix.hash();
  const wm2 = int2.weightMatrix.hash();
  console.log(`WeightMatrix hashes match: ${wm1 === wm2}`);

  const pass = hash1 === hash2 && pt1 === pt2 && wm1 === wm2;
  console.log(pass ? '✓ PASS: V6 deterministic tensor hashing verified' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Main test runner
// -------------------------------------------------------------------------

async function runAllTests() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  Tensor Math Cluster Integration Test Suite                    ║');
  console.log('║  Pi-Geometry · Neural Weights · Inference Ops · Cluster Bind   ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  const tests = [
    { name: 'Pi-Geometry Tensor',       fn: testPiGeometryTensor },
    { name: 'Neural Weight Matrix',     fn: testNeuralWeightMatrix },
    { name: 'TensorValue Spherical',    fn: testTensorValue },
    { name: 'Inference Operations',     fn: testInferenceOperations },
    { name: 'Cognitive Graph',          fn: testCognitiveGraph },
    { name: 'Matrix Operations',        fn: testMatrixOperations },
    { name: 'Inference Calculus',       fn: testInferenceCalculus },
    { name: 'SV3G Tensor Mapping',      fn: testSV3GMapping },
    { name: 'Quantum Cognitive State',  fn: testQuantumCognitiveState },
    { name: 'Validation Metrics',       fn: testValidationMetrics },
    { name: 'Tensor Cluster Integration', fn: testTensorClusterIntegration },
    { name: 'Full Cluster + Tensor',    fn: testFullClusterIntegration },
    { name: 'V6 Deterministic Hashing', fn: testDeterministicTensorHashing },
  ];

  const results = [];
  for (const test of tests) {
    try {
      const pass = await test.fn();
      results.push({ name: test.name, pass });
    } catch (e) {
      console.log(`✗ EXCEPTION in ${test.name}: ${e.message}`);
      console.log(`  ${e.stack?.split('\n')[1]?.trim() || ''}`);
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

runAllTests().then(allPassed => process.exit(allPassed ? 0 : 1));
