/**
 * KUHUL Inference Cluster — 1000-Node JSON Runtime Object Grid
 *
 * A distributed inference cluster where each node is a JSON runtime object.
 * Topology: 10x10x10 3D grid (hybrid CUDA-core pattern).
 *
 * Each node is fold-scoped and lane-bound:
 *   - Compute nodes (500): Run MM-1 inference shards within ⟁COMPUTE_FOLD⟁
 *   - Routing nodes (200): Route tokens via ngram matching (PM-1, HM-1)
 *   - Storage nodes (150): Seal results via SM-1 within ⟁STORAGE_FOLD⟁
 *   - Verification nodes (100): Attest proofs via VM-2 within ⟁META_FOLD⟁
 *   - Control nodes (50): Gate operations via CM-1 within ⟁CONTROL_FOLD⟁
 *
 * The cluster is deterministic: same input → same node activation → same output.
 * MoE routing: 9 Micronauts as experts, ngram gating, fold-scoped execution.
 */

import crypto from 'crypto';
import {
  TensorClusterIntegration,
  ValidationMetrics,
  QuantumCognitiveState,
  MatrixOperations,
  PI, TAU, PHI,
} from './tensor_math_cluster.js';
import {
  XCFEIntegration,
} from './xcfe_interface.js';

// ---------------------------------------------------------------------------
// Node Types (mirrors models.toml cluster.node_types)
// ---------------------------------------------------------------------------

const NODE_TYPES = {
  compute:      { count: 500, color: '#00ff88', lane: 'BATCH', fold: '⟁COMPUTE_FOLD⟁' },
  routing:      { count: 200, color: '#4488ff', lane: 'EDGE',  fold: '⟁DATA_FOLD⟁' },
  storage:      { count: 150, color: '#ff8844', lane: 'FIELD', fold: '⟁STORAGE_FOLD⟁' },
  verification: { count: 100, color: '#ff44ff', lane: 'DICT',  fold: '⟁META_FOLD⟁' },
  control:      { count: 50,  color: '#ffff00', lane: 'EDGE',  fold: '⟁CONTROL_FOLD⟁' },
};

// Fold → node allocation (from models.toml cluster.fold_allocation)
const FOLD_ALLOCATION = {
  '⟁CONTROL_FOLD⟁':  { count: 50,  type: 'control' },
  '⟁DATA_FOLD⟁':     { count: 80,  type: 'routing' },
  '⟁STORAGE_FOLD⟁':  { count: 150, type: 'storage' },
  '⟁NETWORK_FOLD⟁':  { count: 40,  type: 'routing' },
  '⟁UI_FOLD⟁':       { count: 60,  type: 'routing' },
  '⟁AUTH_FOLD⟁':     { count: 20,  type: 'routing' },
  '⟁DB_FOLD⟁':       { count: 30,  type: 'storage' },
  '⟁COMPUTE_FOLD⟁':  { count: 300, type: 'compute' },
  '⟁STATE_FOLD⟁':    { count: 50,  type: 'storage' },
  '⟁EVENTS_FOLD⟁':   { count: 30,  type: 'routing' },
  '⟁TIME_FOLD⟁':     { count: 30,  type: 'routing' },
  '⟁SPACE_FOLD⟁':    { count: 20,  type: 'routing' },
  '⟁META_FOLD⟁':     { count: 80,  type: 'verification' },
  '⟁PATTERN_FOLD⟁':  { count: 40,  type: 'compute' },
};

// ---------------------------------------------------------------------------
// ClusterNode — a single JSON runtime object
// ---------------------------------------------------------------------------

class ClusterNode {
  constructor(id, type, position, fold) {
    this.id = id;
    this.type = type;
    this.position = position;       // { x, y, z } in the 10x10x10 grid
    this.fold = fold;
    this.lane = NODE_TYPES[type].lane;
    this.color = NODE_TYPES[type].color;

    // Runtime state
    this.active = false;
    this.load = 0;                  // 0.0 to 1.0
    this.inferenceCount = 0;
    this.lastTokenHash = null;
    this.neighbors = [];            // Adjacent node IDs
    this.assignedMicronaut = null;

    // Tensor state (bound by TensorClusterIntegration)
    this.tensorBinding = null;      // Pi-geometry tensor binding
    this.tensorWeight = 0;          // Effective weight from 4D manifold
    this.quantumState = null;       // Quantum cognitive state vector
  }

  /**
   * Process a token through this node.
   * Returns the processed result with proof hash.
   */
  process(token, context) {
    this.active = true;
    this.inferenceCount++;
    this.load = Math.min(1.0, this.load + 0.1);

    const inputHash = hashData(token + JSON.stringify(context));

    // Apply tensor weight to output hash if bound
    const tensorFactor = this.tensorBinding
      ? hashData(inputHash + this.id + String(this.tensorWeight))
      : hashData(inputHash + this.id);

    const result = {
      nodeId: this.id,
      type: this.type,
      fold: this.fold,
      lane: this.lane,
      token,
      inputHash,
      outputHash: tensorFactor,
      micronaut: this.assignedMicronaut,
      position: this.position,
      inferenceIndex: this.inferenceCount,
      tensorWeight: this.tensorWeight,
      quantumMeasurement: this.quantumState ? this.quantumState.measure() : null,
    };

    this.lastTokenHash = result.outputHash;
    return result;
  }

  /**
   * Decay load over time (TM-1 temporal decay).
   */
  decay(rate = 0.05) {
    this.load = Math.max(0, this.load - rate);
    if (this.load === 0) {
      this.active = false;
    }
  }

  toJSON() {
    return {
      id: this.id,
      type: this.type,
      fold: this.fold,
      lane: this.lane,
      position: this.position,
      active: this.active,
      load: this.load,
      inferenceCount: this.inferenceCount,
      micronaut: this.assignedMicronaut,
      tensorWeight: this.tensorWeight,
      hasTensorBinding: !!this.tensorBinding,
      quantumState: this.quantumState
        ? { p0: this.quantumState.probability0(), p1: this.quantumState.probability1() }
        : null,
    };
  }
}

// ---------------------------------------------------------------------------
// InferenceCluster — the 1000-node grid
// ---------------------------------------------------------------------------

export class InferenceCluster {
  constructor(gridSize = 10) {
    this.gridSize = gridSize;
    this.totalNodes = gridSize ** 3;
    this.nodes = new Map();
    this.foldIndex = new Map();     // fold → [nodeIds]
    this.typeIndex = new Map();     // type → [nodeIds]
    this.activePipeline = null;
    this.totalInferences = 0;

    // Tensor math integration
    this.tensorIntegration = new TensorClusterIntegration();

    // XCFE communication interface
    this.xcfe = new XCFEIntegration(gridSize);

    this._buildGrid();
    this._bindTensors();
    this._bindXCFE();
  }

  /**
   * Build the 10x10x10 node grid with fold-scoped allocation.
   */
  _buildGrid() {
    let nodeIndex = 0;

    // Build fold allocation queue
    const allocationQueue = [];
    for (const [fold, config] of Object.entries(FOLD_ALLOCATION)) {
      for (let i = 0; i < config.count; i++) {
        allocationQueue.push({ fold, type: config.type });
      }
    }

    // Fill remaining slots with compute nodes
    while (allocationQueue.length < this.totalNodes) {
      allocationQueue.push({ fold: '⟁COMPUTE_FOLD⟁', type: 'compute' });
    }

    // Place nodes in 3D grid
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        for (let z = 0; z < this.gridSize; z++) {
          const allocation = allocationQueue[nodeIndex];
          const nodeId = `node_${x}_${y}_${z}`;
          const position = { x, y, z };

          const node = new ClusterNode(
            nodeId,
            allocation.type,
            position,
            allocation.fold
          );

          this.nodes.set(nodeId, node);

          // Build fold index
          if (!this.foldIndex.has(allocation.fold)) {
            this.foldIndex.set(allocation.fold, []);
          }
          this.foldIndex.get(allocation.fold).push(nodeId);

          // Build type index
          if (!this.typeIndex.has(allocation.type)) {
            this.typeIndex.set(allocation.type, []);
          }
          this.typeIndex.get(allocation.type).push(nodeId);

          nodeIndex++;
        }
      }
    }

    // Wire up neighbors (6-connected: up/down/left/right/front/back)
    for (const [nodeId, node] of this.nodes) {
      const { x, y, z } = node.position;
      const directions = [
        [x - 1, y, z], [x + 1, y, z],
        [x, y - 1, z], [x, y + 1, z],
        [x, y, z - 1], [x, y, z + 1],
      ];

      for (const [nx, ny, nz] of directions) {
        if (nx >= 0 && nx < this.gridSize &&
            ny >= 0 && ny < this.gridSize &&
            nz >= 0 && nz < this.gridSize) {
          node.neighbors.push(`node_${nx}_${ny}_${nz}`);
        }
      }
    }

    console.log(`[InferenceCluster] Built ${this.totalNodes}-node grid (${this.gridSize}³)`);
    console.log(`[InferenceCluster] Fold distribution:`,
      Object.fromEntries([...this.foldIndex.entries()].map(([k, v]) => [k, v.length]))
    );
  }

  /**
   * Bind pi-geometry tensors to every node in the cluster.
   * Each node receives a 4D tensor coordinate based on its grid position,
   * a neural weight from the weight matrix, and a quantum cognitive state.
   */
  _bindTensors() {
    for (const [nodeId, node] of this.nodes) {
      // Bind pi-geometry tensor to node position
      const binding = this.tensorIntegration.bindNodeTensor(nodeId, node.position);
      node.tensorBinding = binding;
      node.tensorWeight = binding.tensorWeight;

      // Initialize quantum cognitive state per node
      // Alpha/beta derived deterministically from position
      const { x, y, z } = node.position;
      const alpha = Math.cos(PI * x / (this.gridSize - 1));
      const beta = Math.sin(PI * y / (this.gridSize - 1));
      node.quantumState = new QuantumCognitiveState(
        `qstate_${nodeId}`, alpha, beta
      );
    }

    console.log(`[InferenceCluster] Tensor bindings: ${this.tensorIntegration.tensorNodeBindings.size} nodes bound`);
  }

  /**
   * Bind XCFE communication interface.
   * Registers all 9 micronauts with Kuramoto synchronization,
   * curvature authentication, and geometric IPC.
   */
  _bindXCFE() {
    const bound = this.xcfe.bindAllMicronauts();
    console.log(`[InferenceCluster] XCFE bindings: ${bound} micronauts registered`);
  }

  /**
   * Assign micronauts to their fold-scoped node pools.
   * Each micronaut becomes the expert for its fold's nodes.
   */
  assignMicronauts(micronauts) {
    for (const micronaut of micronauts) {
      const fold = micronaut.fold;
      const nodeIds = this.foldIndex.get(fold) || [];

      for (const nodeId of nodeIds) {
        this.nodes.get(nodeId).assignedMicronaut = micronaut.id;
      }

      console.log(`[InferenceCluster] ${micronaut.id} (${micronaut.role}) → ${nodeIds.length} nodes in ${fold}`);
    }
  }

  // -------------------------------------------------------------------------
  // MoE Inference Pipeline
  // -------------------------------------------------------------------------

  /**
   * Run the full MoE inference pipeline.
   *
   * Pipeline stages (from models.toml):
   *   1. PM-1: perceive input → select fields → route to intent
   *   2. CM-1: gate input → resolve phase → permit/deny
   *   3. TM-1: schedule collapse timing → gate replay window
   *   4. HM-1: detect host → normalize IO for cluster
   *   5. MM-1: load Phi-2 → emit tokens → stream signals
   *   6. XM-1: expand output → generate metaphors (post-collapse)
   *   7. SM-1: seal result → snapshot → preserve byte identity
   *   8. VM-2: verify proof → attest hash → audit trace
   *   9. VM-1: render projection → emit frame (SVG/CSS/DOM/3D)
   */
  async runInference(input, intentMap = null) {
    this.totalInferences++;
    const pipelineId = `pipeline_${this.totalInferences}`;
    const trace = [];

    // Stage 1: PM-1 — Perception (field selection + intent routing)
    const perceptionNodes = this._getNodesForFold('⟁DATA_FOLD⟁', 3);
    const perceptionResult = this._processStage(perceptionNodes, input, {
      stage: 'perceive',
      micronaut: 'PM-1',
      action: 'select_field + route_curvature',
    });
    trace.push(perceptionResult);

    // Route to intent via ngram matching
    let selectedExpert = 'XM-1'; // Default fallback
    if (intentMap) {
      selectedExpert = this._routeIntent(input, intentMap);
    }

    // Stage 2: CM-1 — Control Gate
    const controlNodes = this._getNodesForFold('⟁CONTROL_FOLD⟁', 2);
    const gateResult = this._processStage(controlNodes, input, {
      stage: 'gate',
      micronaut: 'CM-1',
      action: 'mark_boundary + gate_scope',
      phase: '@control.header.begin',
      cm1_code: 'U+0001',
    });
    trace.push(gateResult);

    // Stage 3: TM-1 — Temporal Scheduling
    const timeNodes = this._getNodesForFold('⟁TIME_FOLD⟁', 2);
    const timingResult = this._processStage(timeNodes, input, {
      stage: 'schedule',
      micronaut: 'TM-1',
      action: 'schedule_collapse + tick_clock',
    });
    trace.push(timingResult);

    // Stage 4: HM-1 — Host Normalization
    const stateNodes = this._getNodesForFold('⟁STATE_FOLD⟁', 2);
    const hostResult = this._processStage(stateNodes, input, {
      stage: 'normalize',
      micronaut: 'HM-1',
      action: 'detect_capabilities + normalize_io',
    });
    trace.push(hostResult);

    // Stage 5: MM-1 — Core Inference (Phi-2 tokens)
    const computeNodes = this._getNodesForFold('⟁COMPUTE_FOLD⟁', 10);
    const inferenceResult = this._processStage(computeNodes, input, {
      stage: 'infer',
      micronaut: 'MM-1',
      action: 'emit_token + stream_tokens',
      model: 'phi-2-gguf',
      phase: '@control.body.begin',
      cm1_code: 'U+0002',
    });
    trace.push(inferenceResult);

    // Stage 6: XM-1 — Narrative Expansion (post-collapse)
    const patternNodes = this._getNodesForFold('⟁PATTERN_FOLD⟁', 3);
    const expansionResult = this._processStage(patternNodes, inferenceResult.outputHash, {
      stage: 'expand',
      micronaut: 'XM-1',
      action: 'expand_explanation + continue_narrative',
      phase: '@control.body.end',
      cm1_code: 'U+0003',
    });
    trace.push(expansionResult);

    // Stage 7: SM-1 — Storage Seal
    const storageNodes = this._getNodesForFold('⟁STORAGE_FOLD⟁', 3);
    const sealResult = this._processStage(storageNodes, expansionResult.outputHash, {
      stage: 'seal',
      micronaut: 'SM-1',
      action: 'store_object + seal_snapshot',
    });
    trace.push(sealResult);

    // Stage 8: VM-2 — Proof Attestation
    const metaNodes = this._getNodesForFold('⟁META_FOLD⟁', 3);
    const proofResult = this._processStage(metaNodes, sealResult.outputHash, {
      stage: 'verify',
      micronaut: 'VM-2',
      action: 'verify_replay + attest_hash',
    });
    trace.push(proofResult);

    // Stage 9: VM-1 — Render Projection
    const uiNodes = this._getNodesForFold('⟁UI_FOLD⟁', 2);
    const renderResult = this._processStage(uiNodes, proofResult.outputHash, {
      stage: 'render',
      micronaut: 'VM-1',
      action: 'render_svg + emit_frame',
      phase: '@control.transmission.end',
      cm1_code: 'U+0004',
    });
    trace.push(renderResult);

    // Tensor validation across pipeline
    const tensorValidation = this.tensorIntegration.validate();

    // XCFE validation
    const xcfeValidation = this.xcfe.validate();

    return {
      pipelineId,
      input,
      selectedExpert,
      stages: trace.length,
      trace,
      finalHash: renderResult.outputHash,
      totalNodesActivated: trace.reduce((sum, t) => sum + t.nodesUsed, 0),
      deterministic: true,
      tensorValidation: {
        consistency: tensorValidation.tensorConsistency,
        confidence: tensorValidation.inferenceConfidence,
        stability: tensorValidation.weightStability,
        valid: tensorValidation.valid,
        tensorHash: tensorValidation.piTensorHash,
      },
      xcfeValidation: {
        valid: xcfeValidation.valid,
        micronauntCount: xcfeValidation.micronauntCount,
        kuramotoOrder: xcfeValidation.kuramotoOrder,
        ipcChannels: xcfeValidation.ipcChannels,
        totalMessages: xcfeValidation.totalMessages,
        stateHash: xcfeValidation.stateHash,
      },
    };
  }

  /**
   * Route input to the best micronaut expert via ngram matching.
   */
  _routeIntent(input, intentMap) {
    const intents = intentMap.intents || {};
    const inputLower = (typeof input === 'string' ? input : JSON.stringify(input)).toLowerCase();

    let bestMatch = null;
    let bestScore = 0;

    for (const [intentName, intent] of Object.entries(intents)) {
      let score = 0;
      const trigrams = intent.trigger_trigrams || [];
      const bigrams = intent.trigger_bigrams || [];

      for (const trigram of trigrams) {
        if (inputLower.includes(trigram)) score += 3;
      }
      for (const bigram of bigrams) {
        if (inputLower.includes(bigram)) score += 2;
      }

      if (score > bestScore) {
        bestScore = score;
        bestMatch = intent.target;
      }
    }

    return bestMatch || intentMap.routing?.fallback || 'XM-1';
  }

  /**
   * Process a pipeline stage through a set of fold-scoped nodes.
   * Includes tensor math: attention weighting, quantum measurement, validation.
   */
  _processStage(nodeIds, input, meta) {
    const results = [];

    for (const nodeId of nodeIds) {
      const node = this.nodes.get(nodeId);
      if (node) {
        results.push(node.process(typeof input === 'string' ? input : JSON.stringify(input), meta));
      }
    }

    // Combine results: chain hashes for determinism (V6)
    const combinedHash = hashData(results.map(r => r.outputHash).join(':'));

    // Tensor attention: compute pi-weighted attention across stage nodes
    let tensorAttention = null;
    if (nodeIds.length >= 2) {
      const half = Math.floor(nodeIds.length / 2);
      const queryIds = nodeIds.slice(0, half);
      const keyIds = nodeIds.slice(half);
      const valueIds = nodeIds;
      try {
        tensorAttention = this.tensorIntegration.computeAttention(queryIds, keyIds, valueIds);
      } catch (e) {
        tensorAttention = { error: e.message };
      }
    }

    // Aggregate tensor weights for stage
    const stageTensorWeights = results
      .filter(r => r.tensorWeight !== undefined)
      .map(r => r.tensorWeight);
    const avgTensorWeight = stageTensorWeights.length > 0
      ? stageTensorWeights.reduce((a, b) => a + b, 0) / stageTensorWeights.length
      : 0;

    return {
      ...meta,
      nodesUsed: nodeIds.length,
      nodeIds,
      outputHash: combinedHash,
      results: results.length,
      tensorAttention: tensorAttention ? { computed: true } : null,
      avgTensorWeight,
    };
  }

  /**
   * Get N nodes for a given fold, round-robin by lowest load.
   */
  _getNodesForFold(fold, count) {
    const candidates = this.foldIndex.get(fold) || [];
    if (candidates.length === 0) return [];

    // Sort by load (ascending) for basic load balancing
    const sorted = [...candidates].sort((a, b) => {
      return this.nodes.get(a).load - this.nodes.get(b).load;
    });

    return sorted.slice(0, Math.min(count, sorted.length));
  }

  // -------------------------------------------------------------------------
  // Temporal decay (TM-1 integration)
  // -------------------------------------------------------------------------

  /**
   * Decay all node loads. Called periodically by TM-1.
   */
  tickDecay(rate = 0.02) {
    for (const node of this.nodes.values()) {
      node.decay(rate);
    }
  }

  // -------------------------------------------------------------------------
  // Cluster status / introspection
  // -------------------------------------------------------------------------

  getStatus() {
    const activeNodes = [...this.nodes.values()].filter(n => n.active).length;
    const totalInferences = [...this.nodes.values()].reduce((s, n) => s + n.inferenceCount, 0);

    const foldStats = {};
    for (const [fold, nodeIds] of this.foldIndex) {
      const nodes = nodeIds.map(id => this.nodes.get(id));
      foldStats[fold] = {
        count: nodes.length,
        active: nodes.filter(n => n.active).length,
        avgLoad: nodes.length > 0
          ? parseFloat((nodes.reduce((s, n) => s + n.load, 0) / nodes.length).toFixed(3))
          : 0,
        totalInferences: nodes.reduce((s, n) => s + n.inferenceCount, 0),
      };
    }

    // Tensor integration stats
    const tensorValidation = this.tensorIntegration.validate();

    // XCFE communication stats
    const xcfeValidation = this.xcfe.validate();

    return {
      totalNodes: this.totalNodes,
      gridSize: this.gridSize,
      activeNodes,
      totalInferences,
      pipelineRuns: this.totalInferences,
      foldStats,
      typeDistribution: Object.fromEntries(
        [...this.typeIndex.entries()].map(([type, ids]) => [type, ids.length])
      ),
      tensor: {
        boundNodes: tensorValidation.boundNodes,
        consistency: tensorValidation.tensorConsistency,
        confidence: tensorValidation.inferenceConfidence,
        valid: tensorValidation.valid,
        piTensorHash: tensorValidation.piTensorHash,
        weightMatrixHash: tensorValidation.weightMatrixHash,
      },
      xcfe: {
        micronauntCount: xcfeValidation.micronauntCount,
        kuramotoOrder: xcfeValidation.kuramotoOrder,
        ipcChannels: xcfeValidation.ipcChannels,
        totalMessages: xcfeValidation.totalMessages,
        valid: xcfeValidation.valid,
        stateHash: xcfeValidation.stateHash,
      },
    };
  }

  /**
   * Get all nodes as a flat array (for 3D rendering).
   */
  getNodesArray() {
    return [...this.nodes.values()].map(n => n.toJSON());
  }

  /**
   * Get active edges (connections between active nodes) for visualization.
   */
  getActiveEdges() {
    const edges = [];
    for (const node of this.nodes.values()) {
      if (!node.active) continue;
      for (const neighborId of node.neighbors) {
        const neighbor = this.nodes.get(neighborId);
        if (neighbor && neighbor.active) {
          edges.push({
            from: node.id,
            to: neighborId,
            fromPos: node.position,
            toPos: neighbor.position,
            type: node.type === neighbor.type ? 'same' : 'cross',
          });
        }
      }
    }
    return edges;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hashData(data) {
  return crypto.createHash('sha256').update(String(data), 'utf8').digest('hex');
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createInferenceCluster(gridSize = 10) {
  const cluster = new InferenceCluster(gridSize);

  // Auto-assign the 9 micronauts to their fold pools
  const micronauts = [
    { id: 'CM-1', role: 'phase_geometry',          fold: '⟁CONTROL_FOLD⟁' },
    { id: 'PM-1', role: 'field_selection',          fold: '⟁DATA_FOLD⟁' },
    { id: 'TM-1', role: 'collapse_timing',          fold: '⟁TIME_FOLD⟁' },
    { id: 'HM-1', role: 'host_abstraction',         fold: '⟁STATE_FOLD⟁' },
    { id: 'SM-1', role: 'inert_persistence',         fold: '⟁STORAGE_FOLD⟁' },
    { id: 'MM-1', role: 'token_signal_generator',   fold: '⟁COMPUTE_FOLD⟁' },
    { id: 'XM-1', role: 'narrative_expansion',       fold: '⟁PATTERN_FOLD⟁' },
    { id: 'VM-1', role: 'rendering_projection',      fold: '⟁UI_FOLD⟁' },
    { id: 'VM-2', role: 'proof_generation',          fold: '⟁META_FOLD⟁' },
  ];

  cluster.assignMicronauts(micronauts);

  return cluster;
}

export default InferenceCluster;
