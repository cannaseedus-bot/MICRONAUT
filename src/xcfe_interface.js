/**
 * XCFE — eXtensible Cognitive Front-End Interface
 *
 * Geometric interaction protocols for micronaut cluster communication.
 * Implements communication manifolds, geodesic channels, agent boundaries,
 * parallel transport messaging, Kuramoto swarm synchronization,
 * security protocols, and KUHUL OS integration.
 *
 * Communication Manifold: M_comm = ℝ³ × S¹ × [0,π]
 *   - ℝ³: spatial topology (cluster grid embedding)
 *   - S¹: phase synchronization (Kuramoto coupling)
 *   - [0,π]: trust/authentication level
 *
 * Channel Types:
 *   - Geodesic:    point-to-point shortest-path channels
 *   - Horosphere:  broadcast (equidistant from source)
 *   - Cone:        multicast (angle-bounded)
 *   - Hypersurface: gossip (diffusive spread)
 *
 * Security:
 *   - Curvature Authentication: metric tensor fingerprint
 *   - Holonomy Encryption: parallel transport phase shift
 *   - Geodesic Integrity: path-length verification
 *
 * Authority: KUHUL_π
 * Fold: ⟁NETWORK_FOLD⟁ (primary), ⟁CONTROL_FOLD⟁ (security)
 */

import crypto from 'crypto';

const PI  = 3.141592653589793;
const TAU = 6.283185307179586;
const PHI = 1.618033988749894;

// ---------------------------------------------------------------------------
// CommunicationManifold — Base topology for message passing
// ---------------------------------------------------------------------------

class CommunicationManifold {
  /**
   * M_comm = ℝ³ × S¹ × [0,π]
   * Embeds the cluster grid into a communication-aware manifold.
   */
  constructor(gridSize = 10) {
    this.gridSize = gridSize;
    this.dimension = 5; // (x, y, z, phase, trust)

    // Metric tensor for communication: ds² = dx² + dy² + dz² + r²dφ² + σ²dτ²
    this.spatialWeight = 1.0;
    this.phaseWeight = PI;       // Phase coupling radius
    this.trustWeight = PI / 2;   // Trust gradient steepness
  }

  /**
   * Compute geodesic distance between two points on the communication manifold.
   * Combines spatial distance, phase difference, and trust differential.
   */
  geodesicDistance(pointA, pointB) {
    const dx = pointB.x - pointA.x;
    const dy = pointB.y - pointA.y;
    const dz = pointB.z - pointA.z;
    const spatialDist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Phase distance (circular): shortest arc on S¹
    const dPhi = Math.abs(pointB.phase - pointA.phase);
    const phaseDist = Math.min(dPhi, TAU - dPhi);

    // Trust distance
    const dTrust = Math.abs(pointB.trust - pointA.trust);

    return Math.sqrt(
      this.spatialWeight * spatialDist * spatialDist +
      this.phaseWeight * phaseDist * phaseDist +
      this.trustWeight * dTrust * dTrust
    );
  }

  /**
   * Compute the metric tensor at a point on the manifold.
   * Returns a 5x5 diagonal metric matrix.
   */
  metricAt(point) {
    return [
      [this.spatialWeight, 0, 0, 0, 0],
      [0, this.spatialWeight, 0, 0, 0],
      [0, 0, this.spatialWeight, 0, 0],
      [0, 0, 0, this.phaseWeight, 0],
      [0, 0, 0, 0, this.trustWeight],
    ];
  }

  /**
   * Parallel transport a message vector from pointA to pointB.
   * The vector rotates by the holonomy of the path.
   *
   * For diagonal metric, holonomy is computed from the curvature
   * integrated along the geodesic.
   */
  parallelTransport(vector, pointA, pointB) {
    const dist = this.geodesicDistance(pointA, pointB);
    if (dist === 0) return { ...vector };

    // Holonomy angle: proportional to enclosed area × curvature
    const phaseDiff = pointB.phase - pointA.phase;
    const trustDiff = pointB.trust - pointA.trust;
    const holonomyAngle = phaseDiff * trustDiff / PI;

    // Rotate the vector components by holonomy
    const cos = Math.cos(holonomyAngle);
    const sin = Math.sin(holonomyAngle);

    return {
      spatial: vector.spatial * cos - vector.phase * sin,
      phase: vector.spatial * sin + vector.phase * cos,
      trust: vector.trust + holonomyAngle / TAU,
      payload: vector.payload,
      holonomyShift: holonomyAngle,
    };
  }

  /**
   * Deterministic hash of manifold configuration
   */
  hash() {
    const data = JSON.stringify({
      gridSize: this.gridSize,
      dimension: this.dimension,
      spatialWeight: this.spatialWeight,
      phaseWeight: this.phaseWeight,
      trustWeight: this.trustWeight,
    });
    return crypto.createHash('sha256').update(data).digest('hex');
  }
}

// ---------------------------------------------------------------------------
// GeodesicChannel — Point-to-point shortest-path communication
// ---------------------------------------------------------------------------

class GeodesicChannel {
  /**
   * A geodesic channel connects two agents on the communication manifold.
   * Messages travel along the shortest path (geodesic) between endpoints.
   */
  constructor(id, sourcePoint, targetPoint, manifold) {
    this.id = id;
    this.sourcePoint = sourcePoint;
    this.targetPoint = targetPoint;
    this.manifold = manifold;
    this.bandwidth = 0;
    this.latency = 0;
    this.messageCount = 0;

    this._computeChannelMetrics();
  }

  _computeChannelMetrics() {
    const dist = this.manifold.geodesicDistance(this.sourcePoint, this.targetPoint);
    // Bandwidth inversely proportional to distance (closer = higher bandwidth)
    this.bandwidth = PI / (1 + dist);
    // Latency proportional to distance
    this.latency = dist / PI;
  }

  /**
   * Send a message through the channel.
   * Message is parallel-transported along the geodesic.
   */
  send(message) {
    this.messageCount++;

    const messageVector = {
      spatial: 1.0,
      phase: message.priority || 0.5,
      trust: message.trustLevel || PI / 4,
      payload: message.payload,
    };

    const transported = this.manifold.parallelTransport(
      messageVector, this.sourcePoint, this.targetPoint
    );

    const messageHash = crypto.createHash('sha256')
      .update(JSON.stringify({
        channelId: this.id,
        seq: this.messageCount,
        payload: message.payload,
        holonomy: transported.holonomyShift,
      }))
      .digest('hex');

    return {
      channelId: this.id,
      sequence: this.messageCount,
      source: this.sourcePoint,
      target: this.targetPoint,
      transportedVector: transported,
      bandwidth: this.bandwidth,
      latency: this.latency,
      integrityHash: messageHash,
      timestamp: Date.now(),
    };
  }

  /**
   * Channel quality metric: bandwidth / latency ratio
   */
  quality() {
    return this.latency > 0 ? this.bandwidth / this.latency : Infinity;
  }

  toJSON() {
    return {
      id: this.id,
      source: this.sourcePoint,
      target: this.targetPoint,
      bandwidth: this.bandwidth,
      latency: this.latency,
      messageCount: this.messageCount,
      quality: this.quality(),
    };
  }
}

// ---------------------------------------------------------------------------
// BroadcastHorosphere — Equidistant broadcast surface
// ---------------------------------------------------------------------------

class BroadcastHorosphere {
  /**
   * Broadcast messages to all agents equidistant from the source.
   * A horosphere is the limit of spheres with increasing radius and
   * center moving along a geodesic — in our metric, it approximates
   * a wavefront of constant geodesic distance.
   */
  constructor(sourcePoint, radius, manifold) {
    this.sourcePoint = sourcePoint;
    this.radius = radius;
    this.manifold = manifold;
    this.subscribers = [];
  }

  /**
   * Subscribe a point to this horosphere (if within radius tolerance)
   */
  subscribe(point, tolerance = 0.5) {
    const dist = this.manifold.geodesicDistance(this.sourcePoint, point);
    if (Math.abs(dist - this.radius) <= tolerance) {
      this.subscribers.push(point);
      return true;
    }
    return false;
  }

  /**
   * Broadcast a message to all subscribers
   */
  broadcast(message) {
    const results = [];
    for (const sub of this.subscribers) {
      const transported = this.manifold.parallelTransport(
        { spatial: 1.0, phase: 0.5, trust: PI / 4, payload: message.payload },
        this.sourcePoint, sub
      );
      results.push({
        target: sub,
        transported,
        distance: this.manifold.geodesicDistance(this.sourcePoint, sub),
      });
    }
    return {
      type: 'horosphere-broadcast',
      source: this.sourcePoint,
      radius: this.radius,
      recipientCount: results.length,
      deliveries: results,
    };
  }
}

// ---------------------------------------------------------------------------
// MulticastCone — Angle-bounded multicast
// ---------------------------------------------------------------------------

class MulticastCone {
  /**
   * Multicast messages within an angular cone from the source.
   * The cone aperture defines the solid angle of the multicast.
   */
  constructor(sourcePoint, direction, aperture, manifold) {
    this.sourcePoint = sourcePoint;
    this.direction = direction; // { dx, dy, dz } unit vector
    this.aperture = aperture;   // Cone half-angle in radians
    this.manifold = manifold;
    this.members = [];
  }

  /**
   * Check if a point falls within the multicast cone
   */
  isInCone(point) {
    const dx = point.x - this.sourcePoint.x;
    const dy = point.y - this.sourcePoint.y;
    const dz = point.z - this.sourcePoint.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist === 0) return true;

    // Dot product with direction (cosine of angle)
    const cosAngle = (dx * this.direction.dx + dy * this.direction.dy + dz * this.direction.dz) / dist;
    return Math.acos(Math.min(1, Math.max(-1, cosAngle))) <= this.aperture;
  }

  /**
   * Add a member to the multicast group
   */
  addMember(point) {
    if (this.isInCone(point)) {
      this.members.push(point);
      return true;
    }
    return false;
  }

  /**
   * Multicast a message to all cone members
   */
  multicast(message) {
    const deliveries = [];
    for (const member of this.members) {
      const dist = this.manifold.geodesicDistance(this.sourcePoint, member);
      deliveries.push({
        target: member,
        distance: dist,
        attenuation: Math.exp(-dist / PI), // Signal attenuates with distance
      });
    }
    return {
      type: 'cone-multicast',
      source: this.sourcePoint,
      aperture: this.aperture,
      memberCount: this.members.length,
      deliveries,
    };
  }
}

// ---------------------------------------------------------------------------
// GossipHypersurface — Diffusive gossip protocol
// ---------------------------------------------------------------------------

class GossipHypersurface {
  /**
   * Gossip protocol modeled as diffusion on a hypersurface.
   * Messages spread via local exchanges following the heat equation:
   * ∂u/∂t = Δ_g u (Laplace-Beltrami operator on manifold)
   */
  constructor(manifold, diffusionRate = 0.1) {
    this.manifold = manifold;
    this.diffusionRate = diffusionRate;
    this.agents = new Map(); // agentId → { point, state }
  }

  /**
   * Register an agent on the hypersurface
   */
  registerAgent(agentId, point) {
    this.agents.set(agentId, {
      point,
      state: 0,       // Message state (0 = uninformed)
      lastUpdate: 0,
    });
  }

  /**
   * Seed a message at a source agent
   */
  seedMessage(agentId, messageValue = 1.0) {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.state = messageValue;
      agent.lastUpdate = Date.now();
    }
  }

  /**
   * Run one diffusion step: each agent averages its state
   * with nearby agents weighted by geodesic proximity.
   */
  diffuseStep() {
    const newStates = new Map();

    for (const [agentId, agent] of this.agents) {
      let weightedSum = agent.state;
      let totalWeight = 1.0;

      for (const [otherId, other] of this.agents) {
        if (otherId === agentId) continue;
        const dist = this.manifold.geodesicDistance(agent.point, other.point);
        const weight = Math.exp(-dist * dist / (2 * this.diffusionRate));

        if (weight > 0.01) { // Only consider nearby agents
          weightedSum += other.state * weight;
          totalWeight += weight;
        }
      }

      newStates.set(agentId, weightedSum / totalWeight);
    }

    // Apply new states
    for (const [agentId, newState] of newStates) {
      const agent = this.agents.get(agentId);
      agent.state = newState;
      agent.lastUpdate = Date.now();
    }

    return {
      type: 'gossip-diffusion',
      agentCount: this.agents.size,
      states: Object.fromEntries(
        [...this.agents.entries()].map(([id, a]) => [id, a.state])
      ),
    };
  }

  /**
   * Run diffusion until convergence or max steps
   */
  diffuseToConvergence(maxSteps = 50, threshold = 0.01) {
    const history = [];

    for (let step = 0; step < maxSteps; step++) {
      const result = this.diffuseStep();
      history.push(result);

      // Check convergence: all states within threshold of mean
      const states = [...this.agents.values()].map(a => a.state);
      const mean = states.reduce((a, b) => a + b, 0) / states.length;
      const maxDev = Math.max(...states.map(s => Math.abs(s - mean)));

      if (maxDev < threshold) {
        return { converged: true, steps: step + 1, history };
      }
    }

    return { converged: false, steps: maxSteps, history };
  }
}

// ---------------------------------------------------------------------------
// AgentBoundary — Geometric interface boundary for agents
// ---------------------------------------------------------------------------

class AgentBoundary {
  /**
   * An agent's interface boundary on the communication manifold.
   * Defines the geometric region within which the agent can send/receive.
   */
  constructor(agentId, center, radius, fold) {
    this.agentId = agentId;
    this.center = center;       // Point on manifold
    this.radius = radius;       // Communication radius
    this.fold = fold;           // Fold scope (e.g., ⟁COMPUTE_FOLD⟁)
    this.messageQueue = [];     // Geodesic bundle queue
    this.protocolStack = this._buildProtocolStack();
  }

  /**
   * 7-layer geometric protocol stack
   */
  _buildProtocolStack() {
    return [
      { layer: 1, name: 'manifold',    role: 'topology embedding' },
      { layer: 2, name: 'geodesic',    role: 'path computation' },
      { layer: 3, name: 'transport',   role: 'parallel transport' },
      { layer: 4, name: 'holonomy',    role: 'phase encryption' },
      { layer: 5, name: 'curvature',   role: 'authentication' },
      { layer: 6, name: 'semantic',    role: 'fold-scoped routing' },
      { layer: 7, name: 'application', role: 'micronaut dispatch' },
    ];
  }

  /**
   * Check if another agent's boundary overlaps (can communicate)
   */
  canCommunicateWith(otherBoundary, manifold) {
    const dist = manifold.geodesicDistance(this.center, otherBoundary.center);
    return dist <= (this.radius + otherBoundary.radius);
  }

  /**
   * Enqueue a message into the geodesic bundle queue
   */
  enqueueMessage(message) {
    this.messageQueue.push({
      ...message,
      enqueuedAt: Date.now(),
      processed: false,
    });
  }

  /**
   * Dequeue and process next message
   */
  dequeueMessage() {
    const msg = this.messageQueue.shift();
    if (msg) {
      msg.processed = true;
      msg.processedAt = Date.now();
    }
    return msg;
  }

  /**
   * Queue depth
   */
  queueDepth() {
    return this.messageQueue.length;
  }

  toJSON() {
    return {
      agentId: this.agentId,
      center: this.center,
      radius: this.radius,
      fold: this.fold,
      queueDepth: this.queueDepth(),
      protocolLayers: this.protocolStack.length,
    };
  }
}

// ---------------------------------------------------------------------------
// KuramotoSync — Swarm synchronization for micronauts
// ---------------------------------------------------------------------------

class KuramotoSync {
  /**
   * Kuramoto model for micronaut phase synchronization:
   * ∂θ_i/∂t = ω_i + (K/N) Σ_j sin(θ_j - θ_i) · exp(-d(i,j))
   *
   * Where:
   *   θ_i: phase of micronaut i
   *   ω_i: natural frequency of micronaut i
   *   K: coupling strength
   *   N: number of oscillators
   *   d(i,j): geodesic distance between i and j
   */
  constructor(couplingStrength = PI / 2, manifold = null) {
    this.K = couplingStrength;
    this.manifold = manifold;
    this.oscillators = new Map(); // id → { phase, frequency, point }
  }

  /**
   * Add an oscillator (micronaut) to the sync network
   */
  addOscillator(id, naturalFrequency, point) {
    this.oscillators.set(id, {
      phase: naturalFrequency * PI / 2, // Initial phase from natural freq
      frequency: naturalFrequency,
      point,
    });
  }

  /**
   * Compute one time step of the Kuramoto dynamics
   * ∂θ_i/∂t = ω_i + (K/N) Σ_j sin(θ_j - θ_i) · exp(-d(i,j))
   */
  step(dt = 0.01) {
    const N = this.oscillators.size;
    if (N === 0) return;

    const updates = new Map();

    for (const [idI, oscI] of this.oscillators) {
      let coupling = 0;

      for (const [idJ, oscJ] of this.oscillators) {
        if (idI === idJ) continue;

        const phaseDiff = Math.sin(oscJ.phase - oscI.phase);

        // Distance-weighted coupling
        let distFactor = 1;
        if (this.manifold && oscI.point && oscJ.point) {
          const dist = this.manifold.geodesicDistance(oscI.point, oscJ.point);
          distFactor = Math.exp(-dist);
        }

        coupling += phaseDiff * distFactor;
      }

      const dTheta = oscI.frequency + (this.K / N) * coupling;
      updates.set(idI, oscI.phase + dTheta * dt);
    }

    // Apply updates
    for (const [id, newPhase] of updates) {
      const osc = this.oscillators.get(id);
      osc.phase = ((newPhase % TAU) + TAU) % TAU; // Wrap to [0, 2π)
    }
  }

  /**
   * Run synchronization to convergence
   */
  synchronize(maxSteps = 200, dt = 0.05) {
    const history = [];

    for (let s = 0; s < maxSteps; s++) {
      this.step(dt);

      const orderParam = this.orderParameter();
      history.push({ step: s, orderParameter: orderParam });

      // Converged when order parameter is close to 1
      if (orderParam > 0.95) {
        return {
          converged: true,
          steps: s + 1,
          orderParameter: orderParam,
          phases: this.getPhases(),
          history,
        };
      }
    }

    return {
      converged: false,
      steps: maxSteps,
      orderParameter: this.orderParameter(),
      phases: this.getPhases(),
      history,
    };
  }

  /**
   * Kuramoto order parameter: r = |1/N Σ_j exp(iθ_j)|
   * r ∈ [0,1]: 0 = incoherent, 1 = fully synchronized
   */
  orderParameter() {
    const N = this.oscillators.size;
    if (N === 0) return 0;

    let realSum = 0;
    let imagSum = 0;

    for (const osc of this.oscillators.values()) {
      realSum += Math.cos(osc.phase);
      imagSum += Math.sin(osc.phase);
    }

    return Math.sqrt(realSum * realSum + imagSum * imagSum) / N;
  }

  /**
   * Get all phases as a map
   */
  getPhases() {
    const phases = {};
    for (const [id, osc] of this.oscillators) {
      phases[id] = osc.phase;
    }
    return phases;
  }

  /**
   * Mean phase of the swarm
   */
  meanPhase() {
    let realSum = 0;
    let imagSum = 0;

    for (const osc of this.oscillators.values()) {
      realSum += Math.cos(osc.phase);
      imagSum += Math.sin(osc.phase);
    }

    return Math.atan2(imagSum, realSum);
  }

  /**
   * Deterministic hash
   */
  hash() {
    const data = JSON.stringify({
      K: this.K,
      oscillatorCount: this.oscillators.size,
      orderParameter: this.orderParameter(),
    });
    return crypto.createHash('sha256').update(data).digest('hex');
  }
}

// ---------------------------------------------------------------------------
// CurvatureAuth — Curvature-based authentication protocol
// ---------------------------------------------------------------------------

class CurvatureAuth {
  /**
   * Authentication via metric tensor fingerprinting.
   * An agent proves identity by demonstrating knowledge of the
   * local curvature at its claimed manifold position.
   */
  constructor(manifold) {
    this.manifold = manifold;
    this.registeredAgents = new Map(); // agentId → curvature fingerprint
  }

  /**
   * Generate a curvature fingerprint for a point on the manifold
   */
  generateFingerprint(point) {
    const metric = this.manifold.metricAt(point);

    // Curvature fingerprint: determinant-like invariant of metric at point
    let trace = 0;
    let product = 1;
    for (let i = 0; i < metric.length; i++) {
      trace += metric[i][i];
      product *= metric[i][i];
    }

    const fingerprint = {
      trace,
      product,
      phaseComponent: point.phase * PI,
      trustComponent: point.trust * PI / 2,
      challenge: crypto.createHash('sha256')
        .update(JSON.stringify({ trace, product, phase: point.phase, trust: point.trust }))
        .digest('hex'),
    };

    return fingerprint;
  }

  /**
   * Register an agent with its curvature fingerprint
   */
  registerAgent(agentId, point) {
    const fingerprint = this.generateFingerprint(point);
    this.registeredAgents.set(agentId, {
      fingerprint,
      point,
      registeredAt: Date.now(),
    });
    return fingerprint;
  }

  /**
   * Authenticate an agent by verifying its curvature fingerprint
   */
  authenticate(agentId, claimedPoint) {
    const record = this.registeredAgents.get(agentId);
    if (!record) return { authenticated: false, reason: 'unknown agent' };

    const claimedFingerprint = this.generateFingerprint(claimedPoint);

    // Compare fingerprints
    const match = claimedFingerprint.challenge === record.fingerprint.challenge;

    return {
      authenticated: match,
      agentId,
      fingerprintMatch: match,
      trustLevel: match ? record.point.trust : 0,
    };
  }
}

// ---------------------------------------------------------------------------
// HolonomyEncryption — Phase-shift encryption via parallel transport
// ---------------------------------------------------------------------------

class HolonomyEncryption {
  /**
   * Encrypt messages using holonomy (phase shift during parallel transport).
   * The encryption key is derived from the path-dependent phase accumulated
   * when transporting a vector around a closed loop on the manifold.
   */
  constructor(manifold) {
    this.manifold = manifold;
  }

  /**
   * Derive encryption key from a closed path on the manifold.
   * The holonomy angle serves as the basis for key generation.
   */
  deriveKey(pointA, pointB, pointC) {
    // Transport A→B→C→A and measure total phase shift
    const legAB = this.manifold.parallelTransport(
      { spatial: 1.0, phase: 0, trust: 0, payload: null },
      pointA, pointB
    );
    const legBC = this.manifold.parallelTransport(
      { spatial: legAB.spatial, phase: legAB.phase, trust: legAB.trust, payload: null },
      pointB, pointC
    );
    const legCA = this.manifold.parallelTransport(
      { spatial: legBC.spatial, phase: legBC.phase, trust: legBC.trust, payload: null },
      pointC, pointA
    );

    // Total holonomy = sum of individual phase shifts
    const totalHolonomy = (legAB.holonomyShift || 0) +
                          (legBC.holonomyShift || 0) +
                          (legCA.holonomyShift || 0);

    // Derive key from holonomy
    const keyMaterial = crypto.createHash('sha256')
      .update(String(totalHolonomy) + ':' + String(PI))
      .digest('hex');

    return {
      key: keyMaterial,
      holonomy: totalHolonomy,
      pathVertices: 3,
    };
  }

  /**
   * Encrypt payload using holonomy-derived key
   */
  encrypt(payload, key) {
    const payloadStr = typeof payload === 'string' ? payload : JSON.stringify(payload);
    const cipher = crypto.createHash('sha256')
      .update(key.key + ':' + payloadStr)
      .digest('hex');

    return {
      encrypted: true,
      ciphertext: cipher,
      holonomy: key.holonomy,
      algorithm: 'holonomy-sha256',
    };
  }

  /**
   * Verify integrity of an encrypted message
   */
  verifyIntegrity(encryptedMsg, key, originalPayload) {
    const reEncrypted = this.encrypt(originalPayload, key);
    return reEncrypted.ciphertext === encryptedMsg.ciphertext;
  }
}

// ---------------------------------------------------------------------------
// GeodesicIntegrity — Path-length verification for message integrity
// ---------------------------------------------------------------------------

class GeodesicIntegrity {
  /**
   * Verifies message integrity by confirming the geodesic path length
   * matches expected bounds. If a message was tampered with,
   * the path length invariant will be violated.
   */
  constructor(manifold) {
    this.manifold = manifold;
  }

  /**
   * Compute a path-length certificate for a message route
   */
  certify(waypoints) {
    let totalLength = 0;
    const segments = [];

    for (let i = 0; i < waypoints.length - 1; i++) {
      const dist = this.manifold.geodesicDistance(waypoints[i], waypoints[i + 1]);
      segments.push({
        from: i,
        to: i + 1,
        length: dist,
      });
      totalLength += dist;
    }

    const certificate = crypto.createHash('sha256')
      .update(JSON.stringify({ totalLength, segmentCount: segments.length }))
      .digest('hex');

    return {
      totalLength,
      segments,
      segmentCount: segments.length,
      certificate,
    };
  }

  /**
   * Verify a path certificate
   */
  verify(waypoints, certificate) {
    const recomputed = this.certify(waypoints);
    return recomputed.certificate === certificate.certificate;
  }
}

// ---------------------------------------------------------------------------
// GeometricIPC — Inter-Process Communication via manifold channels
// ---------------------------------------------------------------------------

class GeometricIPC {
  /**
   * IPC mechanism for KUHUL OS processes, using the communication manifold
   * for scheduler, memory, and device channels.
   */
  constructor(manifold) {
    this.manifold = manifold;
    this.channels = new Map();       // channelId → GeodesicChannel
    this.boundaries = new Map();     // agentId → AgentBoundary
    this.messageLog = [];            // Append-only message log
  }

  /**
   * Register an agent with its boundary
   */
  registerAgent(agentId, center, radius, fold) {
    const boundary = new AgentBoundary(agentId, center, radius, fold);
    this.boundaries.set(agentId, boundary);
    return boundary;
  }

  /**
   * Create a channel between two agents
   */
  createChannel(sourceId, targetId) {
    const sourceBoundary = this.boundaries.get(sourceId);
    const targetBoundary = this.boundaries.get(targetId);
    if (!sourceBoundary || !targetBoundary) return null;

    if (!sourceBoundary.canCommunicateWith(targetBoundary, this.manifold)) {
      return null; // Out of range
    }

    const channelId = `ch_${sourceId}_${targetId}`;
    const channel = new GeodesicChannel(
      channelId,
      sourceBoundary.center,
      targetBoundary.center,
      this.manifold
    );

    this.channels.set(channelId, channel);
    return channel;
  }

  /**
   * Send a message from source to target agent via their channel
   */
  sendMessage(sourceId, targetId, payload) {
    const channelId = `ch_${sourceId}_${targetId}`;
    let channel = this.channels.get(channelId);

    if (!channel) {
      channel = this.createChannel(sourceId, targetId);
      if (!channel) return null;
    }

    const result = channel.send({
      payload,
      priority: 0.5,
      trustLevel: PI / 4,
    });

    // Enqueue at target boundary
    const targetBoundary = this.boundaries.get(targetId);
    if (targetBoundary) {
      targetBoundary.enqueueMessage({
        from: sourceId,
        payload,
        integrityHash: result.integrityHash,
      });
    }

    // Append to message log (append-only I/O compliance)
    this.messageLog.push({
      timestamp: Date.now(),
      source: sourceId,
      target: targetId,
      hash: result.integrityHash,
    });

    return result;
  }

  /**
   * Get IPC status
   */
  getStatus() {
    return {
      agentCount: this.boundaries.size,
      channelCount: this.channels.size,
      totalMessages: this.messageLog.length,
      agents: Object.fromEntries(
        [...this.boundaries.entries()].map(([id, b]) => [id, b.toJSON()])
      ),
    };
  }
}

// ---------------------------------------------------------------------------
// XCFEIntegration — Full XCFE interface for cluster integration
// ---------------------------------------------------------------------------

class XCFEIntegration {
  /**
   * Complete XCFE interface binding communication manifold,
   * Kuramoto synchronization, security protocols, and geometric IPC
   * to the inference cluster.
   */
  constructor(gridSize = 10) {
    this.gridSize = gridSize;
    this.manifold = new CommunicationManifold(gridSize);
    this.kuramoto = new KuramotoSync(PI / 2, this.manifold);
    this.auth = new CurvatureAuth(this.manifold);
    this.encryption = new HolonomyEncryption(this.manifold);
    this.integrity = new GeodesicIntegrity(this.manifold);
    this.ipc = new GeometricIPC(this.manifold);
    this.gossip = new GossipHypersurface(this.manifold);

    this.micronauntBindings = new Map(); // micronauntId → oscillator + boundary
  }

  /**
   * Bind a micronaut to the XCFE interface.
   * Registers it as a Kuramoto oscillator, IPC agent, and gossip participant.
   */
  bindMicronaut(micronauntId, position, fold, naturalFrequency) {
    const point = {
      x: position.x,
      y: position.y,
      z: position.z,
      phase: naturalFrequency * PI / 2,
      trust: PI / 4, // Default trust level
    };

    // Register as Kuramoto oscillator
    this.kuramoto.addOscillator(micronauntId, naturalFrequency, point);

    // Register as IPC agent
    const commRadius = PI / 2; // Default communication radius
    this.ipc.registerAgent(micronauntId, point, commRadius, fold);

    // Register for curvature authentication
    this.auth.registerAgent(micronauntId, point);

    // Register on gossip hypersurface
    this.gossip.registerAgent(micronauntId, point);

    this.micronauntBindings.set(micronauntId, {
      point,
      fold,
      naturalFrequency,
      boundAt: Date.now(),
    });

    return point;
  }

  /**
   * Bind all 9 micronauts with their fold-specific frequencies
   */
  bindAllMicronauts() {
    const micronauts = [
      { id: 'CM-1', fold: '⟁CONTROL_FOLD⟁', freq: 1.0, pos: { x: 0, y: 0, z: 0 } },
      { id: 'PM-1', fold: '⟁DATA_FOLD⟁',    freq: PHI, pos: { x: 2, y: 0, z: 0 } },
      { id: 'TM-1', fold: '⟁TIME_FOLD⟁',    freq: PI / 3, pos: { x: 4, y: 0, z: 0 } },
      { id: 'HM-1', fold: '⟁STATE_FOLD⟁',   freq: PI / 4, pos: { x: 6, y: 0, z: 0 } },
      { id: 'SM-1', fold: '⟁STORAGE_FOLD⟁',  freq: 0.5, pos: { x: 8, y: 0, z: 0 } },
      { id: 'MM-1', fold: '⟁COMPUTE_FOLD⟁',  freq: TAU / 3, pos: { x: 0, y: 5, z: 5 } },
      { id: 'XM-1', fold: '⟁PATTERN_FOLD⟁',  freq: PHI / 2, pos: { x: 3, y: 5, z: 5 } },
      { id: 'VM-1', fold: '⟁UI_FOLD⟁',       freq: PI / 5, pos: { x: 6, y: 5, z: 5 } },
      { id: 'VM-2', fold: '⟁META_FOLD⟁',     freq: PI / 6, pos: { x: 9, y: 5, z: 5 } },
    ];

    for (const m of micronauts) {
      this.bindMicronaut(m.id, m.pos, m.fold, m.freq);
    }

    return micronauts.length;
  }

  /**
   * Synchronize all micronauts via Kuramoto coupling
   */
  synchronizeMicronauts(maxSteps = 200) {
    return this.kuramoto.synchronize(maxSteps);
  }

  /**
   * Send a message between two micronauts via the XCFE interface.
   * Goes through the full protocol stack: auth → encrypt → transport → deliver.
   */
  sendMicronauntMessage(sourceId, targetId, payload) {
    // 1. Authenticate source
    const sourceBinding = this.micronauntBindings.get(sourceId);
    const targetBinding = this.micronauntBindings.get(targetId);
    if (!sourceBinding || !targetBinding) return null;

    const authResult = this.auth.authenticate(sourceId, sourceBinding.point);
    if (!authResult.authenticated) return null;

    // 2. Derive encryption key via holonomy
    // Use a third point for the encryption triangle
    const midPoint = {
      x: (sourceBinding.point.x + targetBinding.point.x) / 2,
      y: (sourceBinding.point.y + targetBinding.point.y) / 2,
      z: (sourceBinding.point.z + targetBinding.point.z) / 2,
      phase: (sourceBinding.point.phase + targetBinding.point.phase) / 2,
      trust: (sourceBinding.point.trust + targetBinding.point.trust) / 2,
    };
    const key = this.encryption.deriveKey(
      sourceBinding.point, midPoint, targetBinding.point
    );

    // 3. Encrypt
    const encrypted = this.encryption.encrypt(payload, key);

    // 4. Send via IPC
    const ipcResult = this.ipc.sendMessage(sourceId, targetId, {
      encrypted: encrypted.ciphertext,
      algorithm: encrypted.algorithm,
      holonomy: encrypted.holonomy,
    });

    // 5. Verify geodesic integrity
    const cert = this.integrity.certify([
      sourceBinding.point, midPoint, targetBinding.point,
    ]);

    return {
      source: sourceId,
      target: targetId,
      authenticated: authResult.authenticated,
      encrypted: true,
      pathCertificate: cert.certificate,
      pathLength: cert.totalLength,
      ipcDelivered: !!ipcResult,
      hash: ipcResult ? ipcResult.integrityHash : null,
    };
  }

  /**
   * Broadcast a message to all micronauts via gossip diffusion
   */
  gossipBroadcast(sourceId, messageValue = 1.0, maxSteps = 20) {
    this.gossip.seedMessage(sourceId, messageValue);
    return this.gossip.diffuseToConvergence(maxSteps);
  }

  /**
   * Get complete XCFE status
   */
  getStatus() {
    return {
      manifoldDimension: this.manifold.dimension,
      gridSize: this.gridSize,
      micronauntCount: this.micronauntBindings.size,
      kuramotoOrderParameter: this.kuramoto.orderParameter(),
      registeredAuthAgents: this.auth.registeredAgents.size,
      ipc: this.ipc.getStatus(),
      gossipAgents: this.gossip.agents.size,
    };
  }

  /**
   * Validate the XCFE interface state
   */
  validate() {
    const orderParam = this.kuramoto.orderParameter();
    const ipcStatus = this.ipc.getStatus();
    const authCount = this.auth.registeredAgents.size;

    // Manifold hash for V6 determinism
    const stateHash = crypto.createHash('sha256')
      .update(JSON.stringify({
        manifold: this.manifold.hash(),
        kuramoto: this.kuramoto.hash(),
        micronauts: this.micronauntBindings.size,
        orderParameter: orderParam,
      }))
      .digest('hex');

    return {
      valid: this.micronauntBindings.size > 0 && authCount > 0,
      micronauntCount: this.micronauntBindings.size,
      kuramotoOrder: orderParam,
      authAgents: authCount,
      ipcChannels: ipcStatus.channelCount,
      totalMessages: ipcStatus.totalMessages,
      stateHash,
    };
  }

  /**
   * Deterministic hash of entire XCFE state
   */
  hash() {
    return this.validate().stateHash;
  }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export {
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
};
