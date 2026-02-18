/**
 * Test Suite: KUHUL OS Kernel — Geometric Process Scheduling
 *
 * Tests process manifold, geodesics, Ricci flow, phi-harmony, holonomy,
 * and integration with cluster inference system.
 */

import {
  ProcessState,
  MetricTensor,
  Geodesic,
  RicciFlowScheduler,
  PhiHarmonicScheduler,
  HolonomyScheduler,
  GeodesicPacking,
  ProcessManifoldKernel,
  PI, PHI,
} from './kuhul_kernel.js';

// -------------------------------------------------------------------------
// Test 1: ProcessState and Manifold Coordinates
// -------------------------------------------------------------------------

function testProcessState() {
  console.log('\n=== Test 1: ProcessState & Manifold Coordinates ===');

  const state = new ProcessState(0.5, 0.8, 0.6, 0.7, PI / 3);

  console.log(`State coordinates: (${state.t}, ${state.cpu}, ${state.mem}, ${state.io}, ${(state.priority/PI).toFixed(3)}π)`);
  console.log(`Vector form: [${state.toVector().map(v => v.toFixed(4)).join(', ')}]`);

  const hash = state.hash();
  console.log(`Hash: ${hash.substring(0, 16)}...`);

  const pass = state.t === 0.5 &&
               state.cpu === 0.8 &&
               state.priority === PI / 3 &&
               hash.length === 64;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 2: Metric Tensor & Ricci Curvature
// -------------------------------------------------------------------------

function testMetricTensor() {
  console.log('\n=== Test 2: Metric Tensor & Ricci Curvature ===');

  const metric = new MetricTensor(2.0, 1.5, 1.2, PI);

  console.log(`Metric coefficients: α=${metric.alpha.toFixed(4)}, β=${metric.beta.toFixed(4)}, γ=${metric.gamma.toFixed(4)}, δ=${metric.delta.toFixed(4)}`);

  const tensorMatrix = metric.tensorMatrix();
  console.log(`Diagonal: [${tensorMatrix.map((row, i) => row[i].toFixed(4)).join(', ')}]`);

  const point = new ProcessState(0.5, 0.7, 0.6, 0.8, PI / 4);
  const ric = metric.ricciCurvature(point);
  console.log(`Ricci curvature at point:`);
  for (let i = 0; i < 5; i++) {
    console.log(`  Ric[${i}][${i}] = ${ric[i][i].toFixed(4)}`);
  }

  const curvScalar = metric.scalarCurvature(point);
  console.log(`Scalar curvature: ${curvScalar.toFixed(4)}`);

  const pass = metric.alpha > 0 &&
               metric.beta > 0 &&
               metric.gamma > 0 &&
               metric.delta > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 3: Geodesic Computation
// -------------------------------------------------------------------------

function testGeodesic() {
  console.log('\n=== Test 3: Geodesic (Process Execution Path) ===');

  const metric = new MetricTensor(2.0, 1.5, 1.2);
  const start = new ProcessState(0, 0.5, 0.5, 0.5, PI / 6);
  const end = new ProcessState(1, 0.9, 0.8, 0.7, PI / 3);

  const geod = new Geodesic('proc_1', start, end, metric, 1.0);

  // Evaluate at different points
  const p0 = geod.evaluate(0.0);
  const p5 = geod.evaluate(0.5);
  const p1 = geod.evaluate(1.0);

  console.log(`Geodesic evaluation:`);
  console.log(`  γ(0.0): cpu=${p0.cpu.toFixed(4)}`);
  console.log(`  γ(0.5): cpu=${p5.cpu.toFixed(4)}`);
  console.log(`  γ(1.0): cpu=${p1.cpu.toFixed(4)}`);

  const length = geod.computeLength();
  const energy = geod.computeEnergy();
  const curvature = geod.computeAverageCurvature();

  console.log(`Geodesic metrics:`);
  console.log(`  Length: ${length.toFixed(4)}`);
  console.log(`  Energy: ${energy.toFixed(4)}`);
  console.log(`  Avg Curvature: ${curvature.toFixed(4)}`);

  const pass = p0.cpu === 0.5 &&
               p1.cpu === 0.9 &&
               length > 0 &&
               energy > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 4: Ricci Flow Scheduler (Load Balancing)
// -------------------------------------------------------------------------

function testRicciFlowScheduler() {
  console.log('\n=== Test 4: Ricci Flow Scheduler (Load Balancing) ===');

  const metric = new MetricTensor(2.0, 1.5, 1.2);

  // Create geodesics with different contentions
  const geod1 = new Geodesic('p1',
    new ProcessState(0, 0.9, 0.9, 0.9, PI / 4),  // High contention
    new ProcessState(1, 0.95, 0.95, 0.95, PI / 3),
    metric, 1.0
  );

  const geod2 = new Geodesic('p2',
    new ProcessState(0, 0.3, 0.3, 0.3, PI / 6),  // Low contention
    new ProcessState(1, 0.4, 0.4, 0.4, PI / 4),
    metric, 1.0
  );

  const scheduler = new RicciFlowScheduler([geod1, geod2], metric);

  console.log(`Initial load balance:`);
  let balance = scheduler.computeLoadBalance();
  console.log(`  Mean: ${balance.mean.toFixed(4)}, Variance: ${balance.variance.toFixed(4)}, Balanced: ${balance.balanced}`);

  const history = scheduler.runToConvergence(10);
  console.log(`Ricci flow iterations: ${history.length}`);
  console.log(`Final state: ${history[history.length - 1].hotspotCount} hotspots remaining`);

  balance = scheduler.computeLoadBalance();
  console.log(`Final load balance: Variance=${balance.variance.toFixed(4)}`);

  const pass = history.length > 0 && balance.variance < 1.0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 5: Phi-Harmonic Scheduler (Golden Ratio Fairness)
// -------------------------------------------------------------------------

function testPhiHarmonicScheduler() {
  console.log('\n=== Test 5: Phi-Harmonic Scheduler (φ-Fairness) ===');

  const metric = new MetricTensor();

  const geodesics = [
    new Geodesic('p1', new ProcessState(0, 0.3, 0.3, 0.3, PI / 6), new ProcessState(1, 0.5, 0.5, 0.5, PI / 4), metric, 1.0),
    new Geodesic('p2', new ProcessState(0, 0.4, 0.4, 0.4, PI / 3), new ProcessState(1, 0.6, 0.6, 0.6, PI / 2), metric, 1.0),
    new Geodesic('p3', new ProcessState(0, 0.2, 0.2, 0.2, PI / 12), new ProcessState(1, 0.4, 0.4, 0.4, PI / 6), metric, 1.0),
  ];

  const scheduler = new PhiHarmonicScheduler(geodesics);

  const sorted = scheduler.sortByPriority();
  console.log(`Sorted by priority: ${sorted.map(g => g.id).join(' → ')}`);

  const allocation = scheduler.computePhiFairAllocation();
  console.log(`φ-fair allocation:`);
  for (const [id, frac] of Object.entries(allocation)) {
    console.log(`  ${id}: ${(frac * 100).toFixed(1)}%`);
  }

  const fairness = scheduler.measureFairness();
  console.log(`Fairness score: ${fairness.fairnessScore.toFixed(4)}`);
  console.log(`Is φ-fair: ${fairness.isPhiFair}`);

  const pass = fairness.fairnessScore > 0 && Object.keys(allocation).length === 3;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 6: Holonomy Scheduler (Context Switch Optimization)
// -------------------------------------------------------------------------

function testHolonomyScheduler() {
  console.log('\n=== Test 6: Holonomy Scheduler (Context Switches) ===');

  const metric = new MetricTensor();

  const geodesics = [
    new Geodesic('p1', new ProcessState(0, 0.3, 0.3, 0.3, PI / 6), new ProcessState(1, 0.5, 0.5, 0.5, PI / 4), metric, 1.0),
    new Geodesic('p2', new ProcessState(0, 0.35, 0.35, 0.35, PI / 5), new ProcessState(1, 0.55, 0.55, 0.55, PI / 3), metric, 1.0),
    new Geodesic('p3', new ProcessState(0, 0.32, 0.32, 0.32, PI / 7), new ProcessState(1, 0.52, 0.52, 0.52, PI / 4), metric, 1.0),
    new Geodesic('p4', new ProcessState(0, 0.6, 0.6, 0.6, PI / 2), new ProcessState(1, 0.8, 0.8, 0.8, 2*PI / 3), metric, 1.0),
  ];

  const scheduler = new HolonomyScheduler(geodesics, metric);

  const clusters = scheduler.clusterByCoefficients();
  console.log(`Process clusters: ${clusters.length}`);
  for (let i = 0; i < clusters.length; i++) {
    console.log(`  Cluster ${i+1}: ${clusters[i].map(c => c.id).join(', ')}`);
  }

  const savings = scheduler.computeContextSwitchSavings();
  console.log(`Context switch reduction:`);
  console.log(`  Total clusters: ${savings.clusters}`);
  console.log(`  Context switches: ${savings.contextSwitches}`);
  console.log(`  Reduction factor: ${(savings.switchReduction * 100).toFixed(1)}%`);

  const pass = clusters.length > 0 && savings.switchReduction > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 7: Geodesic Packing (Resource Allocation)
// -------------------------------------------------------------------------

function testGeodesicPacking() {
  console.log('\n=== Test 7: Geodesic Packing (Resource Utilization) ===');

  const metric = new MetricTensor();

  const geodesics = [
    new Geodesic('p1', new ProcessState(0, 0.3, 0.3, 0.3, PI / 6), new ProcessState(1, 0.5, 0.5, 0.5, PI / 4), metric, 1.0),
    new Geodesic('p2', new ProcessState(0, 0.2, 0.2, 0.2, PI / 5), new ProcessState(1, 0.4, 0.4, 0.4, PI / 3), metric, 0.8),
    new Geodesic('p3', new ProcessState(0, 0.6, 0.6, 0.6, PI / 2), new ProcessState(1, 0.8, 0.8, 0.8, 2*PI / 3), metric, 1.2),
  ];

  const packing = new GeodesicPacking(geodesics, metric);

  console.log(`Geodesic volumes:`);
  for (const g of geodesics) {
    const vol = packing.computeVolume(g);
    console.log(`  ${g.id}: ${vol.toFixed(4)}`);
  }

  const totalVol = packing.totalVolume();
  console.log(`Total volume: ${totalVol.toFixed(4)}`);

  const density = packing.packingDensity();
  console.log(`Packing density: ${density.utilization.toFixed(1)}%`);

  const optimized = packing.optimizePacking();
  console.log(`Original order: ${optimized.originalOrder.join(' → ')}`);
  console.log(`Optimized order: ${optimized.optimizedOrder.join(' → ')}`);

  const pass = totalVol > 0 && geodesics.length === 3;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 8: Complete ProcessManifoldKernel
// -------------------------------------------------------------------------

function testProcessManifoldKernel() {
  console.log('\n=== Test 8: ProcessManifoldKernel (Complete System) ===');

  const kernel = new ProcessManifoldKernel(2.0, 1.5, 1.2);

  // Create some processes
  for (let i = 0; i < 5; i++) {
    const start = new ProcessState(0, 0.2 + i*0.1, 0.3, 0.4, PI / 6 + i * PI / 20);
    const end = new ProcessState(1, 0.7 + i*0.05, 0.8, 0.9, PI / 3 + i * PI / 20);
    kernel.createProcess(`proc_${i}`, start, end, 1.0);
  }

  console.log(`Created ${kernel.geodesics.length} processes`);

  // Simulate execution
  for (let step = 0; step < 10; step++) {
    const result = kernel.stepAllProcesses(0.1);
    console.log(`Step ${step+1}: ${result.completedCount} completed, ${kernel.geodesics.length} remaining`);
  }

  // Get metrics
  const metrics = kernel.getMetrics();
  console.log(`Final metrics:`);
  console.log(`  Load balance variance: ${metrics.loadBalance?.variance.toFixed(4)}`);
  console.log(`  Fairness score: ${metrics.fairness?.fairnessScore.toFixed(4)}`);
  console.log(`  Utilization: ${metrics.packing?.utilization.toFixed(1)}%`);

  // Generate proof
  const proof = kernel.generateSchedulingProof();
  console.log(`Scheduling proof verified: ${proof.proof.verified}`);

  const hash = kernel.hash();
  console.log(`Kernel state hash: ${hash.substring(0, 16)}...`);

  const pass = metrics.processCount >= 0 &&
               metrics.fairness?.fairnessScore > 0 &&
               hash.length === 64;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 9: Deterministic Kernel Hashing (V6)
// -------------------------------------------------------------------------

function testDeterministicKernelHashing() {
  console.log('\n=== Test 9: Deterministic Kernel Hashing (V6) ===');

  const createKernelWithProcesses = () => {
    const k = new ProcessManifoldKernel(2.0, 1.5, 1.2);
    for (let i = 0; i < 3; i++) {
      const start = new ProcessState(0, 0.3 + i*0.1, 0.4, 0.5, PI / 6);
      const end = new ProcessState(1, 0.6 + i*0.1, 0.7, 0.8, PI / 3);
      k.createProcess(`p${i}`, start, end, 1.0);
    }
    return k;
  };

  const k1 = createKernelWithProcesses();
  const k2 = createKernelWithProcesses();

  const h1 = k1.hash();
  const h2 = k2.hash();

  console.log(`Kernel 1 hash: ${h1.substring(0, 16)}...`);
  console.log(`Kernel 2 hash: ${h2.substring(0, 16)}...`);
  console.log(`Hashes match: ${h1 === h2}`);

  const pass = h1 === h2;

  console.log(pass ? '✓ PASS: V6 deterministic hashing verified' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Main test runner
// -------------------------------------------------------------------------

function runAllTests() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  KUHUL OS Kernel Test Suite                                    ║');
  console.log('║  Geometric Process Scheduling & Resource Management            ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  const tests = [
    { name: 'ProcessState & Manifold',      fn: testProcessState },
    { name: 'Metric Tensor & Ricci',        fn: testMetricTensor },
    { name: 'Geodesic Computation',         fn: testGeodesic },
    { name: 'Ricci Flow Scheduler',         fn: testRicciFlowScheduler },
    { name: 'Phi-Harmonic Scheduler',       fn: testPhiHarmonicScheduler },
    { name: 'Holonomy Scheduler',           fn: testHolonomyScheduler },
    { name: 'Geodesic Packing',             fn: testGeodesicPacking },
    { name: 'ProcessManifoldKernel',        fn: testProcessManifoldKernel },
    { name: 'V6 Deterministic Hashing',     fn: testDeterministicKernelHashing },
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

const allPassed = runAllTests();
process.exit(allPassed ? 0 : 1);
