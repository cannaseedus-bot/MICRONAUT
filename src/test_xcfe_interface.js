/**
 * Test Suite: XCFE Interface — eXtensible Cognitive Front-End
 *
 * Tests communication manifold, geodesic channels, broadcast/multicast,
 * gossip diffusion, Kuramoto synchronization, security protocols,
 * geometric IPC, and full XCFE integration with cluster inference.
 */

import {
  PI, TAU, PHI,
  CommunicationManifold,
  GeodesicChannel,
  BroadcastHorosphere,
  MulticastCone,
  GossipHypersurface,
  AgentBoundary,
  KuramotoSync,
  CurvatureAuth,
  HolonomyEncryption,
  GeodesicIntegrity,
  GeometricIPC,
  XCFEIntegration,
} from './xcfe_interface.js';

import { InferenceCluster, createInferenceCluster } from './inference_cluster.js';

// -------------------------------------------------------------------------
// Test 1: Communication Manifold
// -------------------------------------------------------------------------

function testCommunicationManifold() {
  console.log('\n=== Test 1: Communication Manifold ===');

  const manifold = new CommunicationManifold(10);

  console.log(`Dimension: ${manifold.dimension} (expected: 5)`);
  console.log(`Grid size: ${manifold.gridSize}`);

  // Geodesic distance
  const pA = { x: 0, y: 0, z: 0, phase: 0, trust: 0 };
  const pB = { x: 3, y: 4, z: 0, phase: PI / 2, trust: PI / 4 };
  const dist = manifold.geodesicDistance(pA, pB);
  console.log(`Distance A→B: ${dist.toFixed(4)}`);

  // Distance should be > 5 (spatial 5 + phase/trust contributions)
  console.log(`Distance > spatial component (5): ${dist > 5}`);

  // Self-distance should be 0
  const selfDist = manifold.geodesicDistance(pA, pA);
  console.log(`Self distance: ${selfDist.toFixed(4)} (expected: 0)`);

  // Metric at a point
  const metric = manifold.metricAt(pA);
  console.log(`Metric diagonal: [${metric.map((r, i) => r[i].toFixed(2)).join(', ')}]`);

  // Parallel transport
  const vector = { spatial: 1.0, phase: 0.5, trust: PI / 4, payload: 'test' };
  const transported = manifold.parallelTransport(vector, pA, pB);
  console.log(`Transported holonomy shift: ${transported.holonomyShift.toFixed(4)}`);
  console.log(`Payload preserved: ${transported.payload === 'test'}`);

  // Hash
  const hash = manifold.hash();
  console.log(`Hash: ${hash.substring(0, 16)}...`);

  const pass = manifold.dimension === 5 &&
               dist > 5 &&
               selfDist === 0 &&
               transported.payload === 'test' &&
               hash.length === 64;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 2: Geodesic Channel
// -------------------------------------------------------------------------

function testGeodesicChannel() {
  console.log('\n=== Test 2: Geodesic Channel (Point-to-Point) ===');

  const manifold = new CommunicationManifold(10);
  const src = { x: 1, y: 1, z: 1, phase: 0, trust: PI / 4 };
  const tgt = { x: 5, y: 5, z: 5, phase: PI / 3, trust: PI / 2 };

  const channel = new GeodesicChannel('ch_test', src, tgt, manifold);

  console.log(`Channel bandwidth: ${channel.bandwidth.toFixed(4)}`);
  console.log(`Channel latency: ${channel.latency.toFixed(4)}`);
  console.log(`Quality: ${channel.quality().toFixed(4)}`);

  // Send a message
  const result = channel.send({
    payload: 'inference request',
    priority: 0.8,
    trustLevel: PI / 3,
  });

  console.log(`Message sequence: ${result.sequence}`);
  console.log(`Integrity hash: ${result.integrityHash.substring(0, 16)}...`);
  console.log(`Transported holonomy: ${result.transportedVector.holonomyShift.toFixed(4)}`);

  // Send another to verify sequence
  const result2 = channel.send({ payload: 'second message' });
  console.log(`Second message seq: ${result2.sequence}`);

  const json = channel.toJSON();
  console.log(`JSON messageCount: ${json.messageCount}`);

  const pass = channel.bandwidth > 0 &&
               channel.latency > 0 &&
               result.sequence === 1 &&
               result2.sequence === 2 &&
               result.integrityHash.length === 64;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 3: Broadcast Horosphere
// -------------------------------------------------------------------------

function testBroadcastHorosphere() {
  console.log('\n=== Test 3: Broadcast Horosphere ===');

  const manifold = new CommunicationManifold(10);
  const source = { x: 5, y: 5, z: 5, phase: 0, trust: PI / 4 };

  const horo = new BroadcastHorosphere(source, 3.0, manifold);

  // Add subscribers at various distances
  const points = [
    { x: 7, y: 5, z: 5, phase: 0.1, trust: PI / 4 },  // Close
    { x: 5, y: 7, z: 6, phase: 0.2, trust: PI / 3 },   // Mid
    { x: 8, y: 5, z: 5, phase: 0, trust: PI / 4 },     // ~dist 3
    { x: 0, y: 0, z: 0, phase: 0, trust: 0 },           // Far
  ];

  let subscribed = 0;
  for (const p of points) {
    if (horo.subscribe(p, 1.5)) subscribed++;
  }
  console.log(`Subscribers: ${subscribed} (of ${points.length} attempted)`);

  // Broadcast
  const result = horo.broadcast({ payload: 'broadcast test' });
  console.log(`Broadcast type: ${result.type}`);
  console.log(`Recipients: ${result.recipientCount}`);

  const pass = result.type === 'horosphere-broadcast' &&
               result.recipientCount === subscribed &&
               subscribed > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 4: Multicast Cone
// -------------------------------------------------------------------------

function testMulticastCone() {
  console.log('\n=== Test 4: Multicast Cone ===');

  const manifold = new CommunicationManifold(10);
  const source = { x: 0, y: 0, z: 0, phase: 0, trust: PI / 4 };

  // Cone pointing in +x direction with π/4 aperture
  const cone = new MulticastCone(source, { dx: 1, dy: 0, dz: 0 }, PI / 4, manifold);

  // Points along +x should be in cone
  const inCone1 = { x: 5, y: 0, z: 0, phase: 0, trust: 0 };
  const inCone2 = { x: 5, y: 1, z: 0, phase: 0, trust: 0 };
  // Point in -x should not be in cone
  const outCone = { x: -5, y: 0, z: 0, phase: 0, trust: 0 };
  // Point at wide angle should not be in cone
  const outCone2 = { x: 1, y: 5, z: 0, phase: 0, trust: 0 };

  console.log(`Point (5,0,0) in cone: ${cone.isInCone(inCone1)}`);
  console.log(`Point (5,1,0) in cone: ${cone.isInCone(inCone2)}`);
  console.log(`Point (-5,0,0) in cone: ${cone.isInCone(outCone)}`);
  console.log(`Point (1,5,0) in cone: ${cone.isInCone(outCone2)}`);

  cone.addMember(inCone1);
  cone.addMember(inCone2);
  cone.addMember(outCone);  // Should fail
  cone.addMember(outCone2); // Should fail

  console.log(`Members: ${cone.members.length} (expected: 2)`);

  const result = cone.multicast({ payload: 'multicast test' });
  console.log(`Multicast type: ${result.type}`);
  console.log(`Deliveries: ${result.memberCount}`);

  // Attenuation should decrease with distance
  if (result.deliveries.length >= 2) {
    console.log(`Attenuation[0]: ${result.deliveries[0].attenuation.toFixed(4)}`);
    console.log(`Attenuation[1]: ${result.deliveries[1].attenuation.toFixed(4)}`);
  }

  const pass = cone.isInCone(inCone1) &&
               cone.isInCone(inCone2) &&
               !cone.isInCone(outCone) &&
               cone.members.length === 2 &&
               result.type === 'cone-multicast';

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 5: Gossip Hypersurface Diffusion
// -------------------------------------------------------------------------

function testGossipHypersurface() {
  console.log('\n=== Test 5: Gossip Hypersurface (Diffusion) ===');

  const manifold = new CommunicationManifold(10);
  const gossip = new GossipHypersurface(manifold, 2.0);

  // Create close-together agents for effective gossip
  gossip.registerAgent('a1', { x: 1, y: 1, z: 1, phase: 0, trust: PI / 4 });
  gossip.registerAgent('a2', { x: 2, y: 1, z: 1, phase: 0.1, trust: PI / 4 });
  gossip.registerAgent('a3', { x: 1, y: 2, z: 1, phase: 0.2, trust: PI / 4 });
  gossip.registerAgent('a4', { x: 2, y: 2, z: 1, phase: 0.1, trust: PI / 4 });

  console.log(`Agents: ${gossip.agents.size}`);

  // Seed message at agent 1
  gossip.seedMessage('a1', 1.0);

  // Check initial states
  console.log(`Initial states:`);
  for (const [id, agent] of gossip.agents) {
    console.log(`  ${id}: ${agent.state.toFixed(4)}`);
  }

  // Run diffusion
  const result = gossip.diffuseToConvergence(30, 0.05);
  console.log(`Converged: ${result.converged}, Steps: ${result.steps}`);

  // All agents should have similar values after convergence
  const finalStates = [...gossip.agents.values()].map(a => a.state);
  const mean = finalStates.reduce((a, b) => a + b, 0) / finalStates.length;
  const maxDev = Math.max(...finalStates.map(s => Math.abs(s - mean)));
  console.log(`Final mean: ${mean.toFixed(4)}, Max deviation: ${maxDev.toFixed(4)}`);

  const pass = gossip.agents.size === 4 &&
               result.converged &&
               maxDev < 0.1;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 6: Agent Boundary & Protocol Stack
// -------------------------------------------------------------------------

function testAgentBoundary() {
  console.log('\n=== Test 6: Agent Boundary & Protocol Stack ===');

  const manifold = new CommunicationManifold(10);

  const b1 = new AgentBoundary('agent1', { x: 2, y: 2, z: 2, phase: 0, trust: PI / 4 }, PI, '⟁COMPUTE_FOLD⟁');
  const b2 = new AgentBoundary('agent2', { x: 4, y: 4, z: 4, phase: 0.5, trust: PI / 3 }, PI, '⟁DATA_FOLD⟁');
  const b3 = new AgentBoundary('agent3', { x: 9, y: 9, z: 9, phase: PI, trust: PI / 2 }, 0.5, '⟁STORAGE_FOLD⟁');

  // Communication checks
  const canComm12 = b1.canCommunicateWith(b2, manifold);
  const canComm13 = b1.canCommunicateWith(b3, manifold);
  console.log(`Agent1 ↔ Agent2 can communicate: ${canComm12}`);
  console.log(`Agent1 ↔ Agent3 can communicate: ${canComm13}`);

  // Protocol stack
  console.log(`Protocol layers: ${b1.protocolStack.length}`);
  for (const layer of b1.protocolStack) {
    console.log(`  Layer ${layer.layer}: ${layer.name} (${layer.role})`);
  }

  // Message queue
  b1.enqueueMessage({ payload: 'msg1' });
  b1.enqueueMessage({ payload: 'msg2' });
  console.log(`Queue depth: ${b1.queueDepth()}`);

  const msg = b1.dequeueMessage();
  console.log(`Dequeued: ${msg.payload}, Processed: ${msg.processed}`);
  console.log(`Queue after dequeue: ${b1.queueDepth()}`);

  const json = b1.toJSON();
  console.log(`JSON fold: ${json.fold}`);

  const pass = b1.protocolStack.length === 7 &&
               canComm12 === true &&
               b1.queueDepth() === 1 &&
               msg.payload === 'msg1' &&
               msg.processed === true;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 7: Kuramoto Synchronization
// -------------------------------------------------------------------------

function testKuramotoSync() {
  console.log('\n=== Test 7: Kuramoto Synchronization ===');

  const manifold = new CommunicationManifold(10);
  const kuramoto = new KuramotoSync(PI, manifold); // Strong coupling

  // Add 9 micronauts as oscillators with nearby positions
  const micronauts = [
    { id: 'CM-1', freq: 1.0, pos: { x: 1, y: 1, z: 1, phase: 0, trust: PI / 4 } },
    { id: 'PM-1', freq: 1.1, pos: { x: 2, y: 1, z: 1, phase: 0, trust: PI / 4 } },
    { id: 'TM-1', freq: 0.9, pos: { x: 1, y: 2, z: 1, phase: 0, trust: PI / 4 } },
    { id: 'HM-1', freq: 1.05, pos: { x: 2, y: 2, z: 1, phase: 0, trust: PI / 4 } },
    { id: 'SM-1', freq: 0.95, pos: { x: 1, y: 1, z: 2, phase: 0, trust: PI / 4 } },
    { id: 'MM-1', freq: 1.02, pos: { x: 2, y: 1, z: 2, phase: 0, trust: PI / 4 } },
    { id: 'XM-1', freq: 0.98, pos: { x: 1, y: 2, z: 2, phase: 0, trust: PI / 4 } },
    { id: 'VM-1', freq: 1.03, pos: { x: 2, y: 2, z: 2, phase: 0, trust: PI / 4 } },
    { id: 'VM-2', freq: 0.97, pos: { x: 1, y: 1, z: 3, phase: 0, trust: PI / 4 } },
  ];

  for (const m of micronauts) {
    kuramoto.addOscillator(m.id, m.freq, m.pos);
  }

  console.log(`Oscillators: ${kuramoto.oscillators.size}`);

  // Initial order parameter (should be relatively low)
  const initialOrder = kuramoto.orderParameter();
  console.log(`Initial order parameter: ${initialOrder.toFixed(4)}`);

  // Synchronize
  const result = kuramoto.synchronize(300, 0.05);
  console.log(`Converged: ${result.converged}`);
  console.log(`Steps: ${result.steps}`);
  console.log(`Final order parameter: ${result.orderParameter.toFixed(4)}`);
  console.log(`Mean phase: ${kuramoto.meanPhase().toFixed(4)}`);

  // Hash
  const hash = kuramoto.hash();
  console.log(`Hash: ${hash.substring(0, 16)}...`);

  const pass = kuramoto.oscillators.size === 9 &&
               result.orderParameter > 0.9;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 8: Curvature Authentication
// -------------------------------------------------------------------------

function testCurvatureAuth() {
  console.log('\n=== Test 8: Curvature Authentication ===');

  const manifold = new CommunicationManifold(10);
  const auth = new CurvatureAuth(manifold);

  // Register agent
  const point = { x: 3, y: 4, z: 5, phase: PI / 3, trust: PI / 2 };
  const fingerprint = auth.registerAgent('agent1', point);
  console.log(`Fingerprint trace: ${fingerprint.trace.toFixed(4)}`);
  console.log(`Fingerprint challenge: ${fingerprint.challenge.substring(0, 16)}...`);

  // Authenticate with correct position
  const authCorrect = auth.authenticate('agent1', point);
  console.log(`Auth correct position: ${authCorrect.authenticated}`);
  console.log(`Trust level: ${authCorrect.trustLevel.toFixed(4)}`);

  // Authenticate with wrong position
  const wrongPoint = { x: 3, y: 4, z: 5, phase: PI / 2, trust: PI / 3 };
  const authWrong = auth.authenticate('agent1', wrongPoint);
  console.log(`Auth wrong position: ${authWrong.authenticated}`);

  // Authenticate unknown agent
  const authUnknown = auth.authenticate('unknown', point);
  console.log(`Auth unknown agent: ${authUnknown.authenticated}`);

  const pass = authCorrect.authenticated === true &&
               authWrong.authenticated === false &&
               authUnknown.authenticated === false &&
               authCorrect.trustLevel > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 9: Holonomy Encryption
// -------------------------------------------------------------------------

function testHolonomyEncryption() {
  console.log('\n=== Test 9: Holonomy Encryption ===');

  const manifold = new CommunicationManifold(10);
  const encryption = new HolonomyEncryption(manifold);

  const pA = { x: 0, y: 0, z: 0, phase: 0, trust: 0 };
  const pB = { x: 3, y: 4, z: 0, phase: PI / 3, trust: PI / 4 };
  const pC = { x: 5, y: 2, z: 3, phase: PI / 2, trust: PI / 3 };

  // Derive key from triangle A→B→C→A
  const key = encryption.deriveKey(pA, pB, pC);
  console.log(`Key: ${key.key.substring(0, 16)}...`);
  console.log(`Holonomy: ${key.holonomy.toFixed(6)}`);
  console.log(`Path vertices: ${key.pathVertices}`);

  // Encrypt a payload
  const payload = { message: 'secret inference data', fold: '⟁COMPUTE_FOLD⟁' };
  const encrypted = encryption.encrypt(payload, key);
  console.log(`Encrypted: ${encrypted.encrypted}`);
  console.log(`Ciphertext: ${encrypted.ciphertext.substring(0, 16)}...`);
  console.log(`Algorithm: ${encrypted.algorithm}`);

  // Verify integrity with correct key + payload
  const isValid = encryption.verifyIntegrity(encrypted, key, payload);
  console.log(`Integrity valid: ${isValid}`);

  // Verify integrity with wrong payload
  const isInvalid = encryption.verifyIntegrity(encrypted, key, { message: 'tampered' });
  console.log(`Tampered integrity: ${isInvalid}`);

  const pass = key.key.length === 64 &&
               encrypted.encrypted === true &&
               isValid === true &&
               isInvalid === false;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 10: Geodesic Integrity
// -------------------------------------------------------------------------

function testGeodesicIntegrity() {
  console.log('\n=== Test 10: Geodesic Integrity ===');

  const manifold = new CommunicationManifold(10);
  const integrity = new GeodesicIntegrity(manifold);

  // Create a waypoint path
  const waypoints = [
    { x: 0, y: 0, z: 0, phase: 0, trust: PI / 4 },
    { x: 3, y: 0, z: 0, phase: PI / 6, trust: PI / 4 },
    { x: 3, y: 4, z: 0, phase: PI / 3, trust: PI / 3 },
    { x: 3, y: 4, z: 5, phase: PI / 2, trust: PI / 2 },
  ];

  // Certify the path
  const cert = integrity.certify(waypoints);
  console.log(`Total path length: ${cert.totalLength.toFixed(4)}`);
  console.log(`Segments: ${cert.segmentCount}`);
  console.log(`Certificate: ${cert.certificate.substring(0, 16)}...`);

  for (const seg of cert.segments) {
    console.log(`  Segment ${seg.from}→${seg.to}: length=${seg.length.toFixed(4)}`);
  }

  // Verify with same waypoints
  const isValid = integrity.verify(waypoints, cert);
  console.log(`Valid path: ${isValid}`);

  // Verify with tampered waypoints
  const tampered = [...waypoints];
  tampered[2] = { x: 4, y: 4, z: 0, phase: PI / 3, trust: PI / 3 };
  const isTampered = integrity.verify(tampered, cert);
  console.log(`Tampered path: ${isTampered}`);

  const pass = cert.segmentCount === 3 &&
               cert.totalLength > 0 &&
               isValid === true &&
               isTampered === false;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 11: Geometric IPC
// -------------------------------------------------------------------------

function testGeometricIPC() {
  console.log('\n=== Test 11: Geometric IPC ===');

  const manifold = new CommunicationManifold(10);
  const ipc = new GeometricIPC(manifold);

  // Register agents
  ipc.registerAgent('CM-1', { x: 1, y: 1, z: 1, phase: 0, trust: PI / 4 }, PI, '⟁CONTROL_FOLD⟁');
  ipc.registerAgent('MM-1', { x: 2, y: 2, z: 2, phase: PI / 4, trust: PI / 3 }, PI, '⟁COMPUTE_FOLD⟁');
  ipc.registerAgent('SM-1', { x: 9, y: 9, z: 9, phase: PI, trust: PI / 2 }, 0.5, '⟁STORAGE_FOLD⟁');

  // Send message between nearby agents
  const result1 = ipc.sendMessage('CM-1', 'MM-1', 'gate_permit');
  console.log(`CM-1 → MM-1: ${result1 ? 'delivered' : 'failed'}`);
  if (result1) {
    console.log(`  Bandwidth: ${result1.bandwidth.toFixed(4)}`);
    console.log(`  Hash: ${result1.integrityHash.substring(0, 16)}...`);
  }

  // Send to far-away agent (may fail if out of range)
  const result2 = ipc.sendMessage('CM-1', 'SM-1', 'seal_request');
  console.log(`CM-1 → SM-1: ${result2 ? 'delivered' : 'out of range'}`);

  // Check message queue at MM-1
  const mm1Boundary = ipc.boundaries.get('MM-1');
  console.log(`MM-1 queue depth: ${mm1Boundary.queueDepth()}`);

  const dequeued = mm1Boundary.dequeueMessage();
  console.log(`Dequeued from MM-1: payload=${dequeued ? dequeued.payload : 'empty'}`);

  // Status
  const status = ipc.getStatus();
  console.log(`IPC Status: ${status.agentCount} agents, ${status.channelCount} channels, ${status.totalMessages} messages`);

  const pass = result1 !== null &&
               result1.integrityHash.length === 64 &&
               mm1Boundary.queueDepth() === 0 && // Was 1 before dequeue
               status.agentCount === 3;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 12: Full XCFE Integration
// -------------------------------------------------------------------------

function testXCFEIntegration() {
  console.log('\n=== Test 12: Full XCFE Integration ===');

  const xcfe = new XCFEIntegration(10);

  // Bind all 9 micronauts
  const count = xcfe.bindAllMicronauts();
  console.log(`Bound micronauts: ${count}`);

  // Status
  const status = xcfe.getStatus();
  console.log(`Manifold dimension: ${status.manifoldDimension}`);
  console.log(`Micronaut count: ${status.micronauntCount}`);
  console.log(`Auth agents: ${status.registeredAuthAgents}`);
  console.log(`Gossip agents: ${status.gossipAgents}`);

  // Synchronize micronauts
  const syncResult = xcfe.synchronizeMicronauts(300);
  console.log(`Kuramoto sync: converged=${syncResult.converged}, order=${syncResult.orderParameter.toFixed(4)}`);

  // Send authenticated+encrypted message between micronauts
  const msgResult = xcfe.sendMicronauntMessage('CM-1', 'MM-1', { command: 'emit_token', model: 'phi-2-gguf' });
  console.log(`Message CM-1→MM-1:`);
  if (msgResult) {
    console.log(`  Authenticated: ${msgResult.authenticated}`);
    console.log(`  Encrypted: ${msgResult.encrypted}`);
    console.log(`  Path length: ${msgResult.pathLength.toFixed(4)}`);
    console.log(`  IPC delivered: ${msgResult.ipcDelivered}`);
  }

  // Gossip broadcast from CM-1
  const gossipResult = xcfe.gossipBroadcast('CM-1', 1.0, 30);
  console.log(`Gossip: converged=${gossipResult.converged}, steps=${gossipResult.steps}`);

  // Validate
  const validation = xcfe.validate();
  console.log(`Valid: ${validation.valid}`);
  console.log(`State hash: ${validation.stateHash.substring(0, 16)}...`);

  const pass = count === 9 &&
               status.micronauntCount === 9 &&
               msgResult !== null &&
               msgResult.authenticated &&
               msgResult.encrypted &&
               validation.valid;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 13: XCFE + Cluster Integration (Full Pipeline)
// -------------------------------------------------------------------------

async function testXCFEClusterIntegration() {
  console.log('\n=== Test 13: XCFE + Cluster Integration ===');

  // Create cluster
  const cluster = createInferenceCluster(10);

  // Create XCFE
  const xcfe = new XCFEIntegration(10);
  xcfe.bindAllMicronauts();

  // Synchronize
  const syncResult = xcfe.synchronizeMicronauts(300);
  console.log(`Kuramoto order: ${syncResult.orderParameter.toFixed(4)}`);

  // Run inference through cluster
  const result = await cluster.runInference('XCFE channel test: compute tensor inference');

  console.log(`Pipeline stages: ${result.stages}`);
  console.log(`Nodes activated: ${result.totalNodesActivated}`);
  console.log(`Tensor valid: ${result.tensorValidation.valid}`);

  // Validate XCFE state
  const xcfeValidation = xcfe.validate();
  console.log(`XCFE valid: ${xcfeValidation.valid}`);

  // Send inter-stage messages
  let messagesSent = 0;
  const stageOrder = ['CM-1', 'PM-1', 'TM-1', 'HM-1', 'MM-1', 'XM-1', 'SM-1', 'VM-2', 'VM-1'];
  for (let i = 0; i < stageOrder.length - 1; i++) {
    const msg = xcfe.sendMicronauntMessage(stageOrder[i], stageOrder[i + 1], {
      stageIndex: i,
      hash: result.trace[i]?.outputHash,
    });
    if (msg && msg.ipcDelivered) messagesSent++;
  }
  console.log(`Inter-stage messages sent: ${messagesSent}/${stageOrder.length - 1}`);

  const pass = result.stages === 9 &&
               result.tensorValidation.valid &&
               xcfeValidation.valid &&
               messagesSent > 0;

  console.log(pass ? '✓ PASS' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Test 14: Deterministic XCFE Hashing (V6)
// -------------------------------------------------------------------------

function testDeterministicXCFEHashing() {
  console.log('\n=== Test 14: Deterministic XCFE Hashing (V6) ===');

  const createXCFE = () => {
    const xcfe = new XCFEIntegration(10);
    xcfe.bindAllMicronauts();
    return xcfe;
  };

  const xcfe1 = createXCFE();
  const xcfe2 = createXCFE();

  const h1 = xcfe1.hash();
  const h2 = xcfe2.hash();

  console.log(`XCFE 1 hash: ${h1.substring(0, 16)}...`);
  console.log(`XCFE 2 hash: ${h2.substring(0, 16)}...`);
  console.log(`Hashes match: ${h1 === h2}`);

  // Manifold hashes
  const mh1 = xcfe1.manifold.hash();
  const mh2 = xcfe2.manifold.hash();
  console.log(`Manifold hashes match: ${mh1 === mh2}`);

  // Kuramoto hashes (before sync, should match)
  const kh1 = xcfe1.kuramoto.hash();
  const kh2 = xcfe2.kuramoto.hash();
  console.log(`Kuramoto hashes match: ${kh1 === kh2}`);

  const pass = h1 === h2 && mh1 === mh2 && kh1 === kh2;
  console.log(pass ? '✓ PASS: V6 deterministic XCFE hashing verified' : '✗ FAIL');
  return pass;
}

// -------------------------------------------------------------------------
// Main test runner
// -------------------------------------------------------------------------

async function runAllTests() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  XCFE Interface Test Suite                                      ║');
  console.log('║  Communication Manifold · Kuramoto · Security · Cluster Bind    ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');

  const tests = [
    { name: 'Communication Manifold',       fn: testCommunicationManifold },
    { name: 'Geodesic Channel',             fn: testGeodesicChannel },
    { name: 'Broadcast Horosphere',         fn: testBroadcastHorosphere },
    { name: 'Multicast Cone',              fn: testMulticastCone },
    { name: 'Gossip Hypersurface',          fn: testGossipHypersurface },
    { name: 'Agent Boundary & Protocol',    fn: testAgentBoundary },
    { name: 'Kuramoto Synchronization',     fn: testKuramotoSync },
    { name: 'Curvature Authentication',     fn: testCurvatureAuth },
    { name: 'Holonomy Encryption',          fn: testHolonomyEncryption },
    { name: 'Geodesic Integrity',           fn: testGeodesicIntegrity },
    { name: 'Geometric IPC',               fn: testGeometricIPC },
    { name: 'Full XCFE Integration',        fn: testXCFEIntegration },
    { name: 'XCFE + Cluster Integration',   fn: testXCFEClusterIntegration },
    { name: 'V6 Deterministic Hashing',     fn: testDeterministicXCFEHashing },
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
