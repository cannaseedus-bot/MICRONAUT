#!/usr/bin/env node
/**
 * Test Runner — Inference Cluster Simulation
 *
 * Runs the 1000-node cluster through a battery of test prompts,
 * simulates LLM inference (stub mode), and writes test_result.json
 * with full pipeline traces, node statistics, and inference metrics.
 *
 * Usage:
 *   node src/test_inference_cluster.js
 *
 * Output:
 *   test_result.json — Full test results with simulated LLM data
 */

import { createInferenceCluster } from './inference_cluster.js';
import { Phi2GGUFLoader } from './phi2_gguf_loader.js';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Test prompts — one per micronaut domain to exercise all MoE experts
// ---------------------------------------------------------------------------

const TEST_PROMPTS = [
  {
    id: 'test_control',
    prompt: 'mark phase boundary and open scope gate for downstream processing',
    expectedExpert: 'CM-1',
    expectedFold: '⟁CONTROL_FOLD⟁',
    category: 'control',
  },
  {
    id: 'test_perception',
    prompt: 'select input field from the observable stream and filter noise',
    expectedExpert: 'PM-1',
    expectedFold: '⟁DATA_FOLD⟁',
    category: 'perception',
  },
  {
    id: 'test_temporal',
    prompt: 'schedule collapse request and gate replay window for verification',
    expectedExpert: 'TM-1',
    expectedFold: '⟁TIME_FOLD⟁',
    category: 'temporal',
  },
  {
    id: 'test_host',
    prompt: 'detect host capability and normalize io interface for the platform',
    expectedExpert: 'HM-1',
    expectedFold: '⟁STATE_FOLD⟁',
    category: 'host',
  },
  {
    id: 'test_storage',
    prompt: 'store object bytes and seal storage snapshot with byte identity',
    expectedExpert: 'SM-1',
    expectedFold: '⟁STORAGE_FOLD⟁',
    category: 'storage',
  },
  {
    id: 'test_inference',
    prompt: 'emit token signal from the model and stream token signals',
    expectedExpert: 'MM-1',
    expectedFold: '⟁COMPUTE_FOLD⟁',
    category: 'inference',
  },
  {
    id: 'test_expansion',
    prompt: 'expand explanation stream and generate narrative metaphor',
    expectedExpert: 'XM-1',
    expectedFold: '⟁PATTERN_FOLD⟁',
    category: 'expansion',
  },
  {
    id: 'test_render',
    prompt: 'render svg projection and emit render frame to the surface',
    expectedExpert: 'VM-1',
    expectedFold: '⟁UI_FOLD⟁',
    category: 'render',
  },
  {
    id: 'test_verify',
    prompt: 'verify replay identity and attest hash binding for proof',
    expectedExpert: 'VM-2',
    expectedFold: '⟁META_FOLD⟁',
    category: 'verification',
  },
  {
    id: 'test_ambiguous',
    prompt: 'help me understand how the system works please',
    expectedExpert: 'XM-1',
    expectedFold: '⟁PATTERN_FOLD⟁',
    category: 'fallback',
  },
];

// ---------------------------------------------------------------------------
// Intent map (loaded from brains or inline for testing)
// ---------------------------------------------------------------------------

function loadIntentMap() {
  const intentPath = path.resolve(__dirname, '../micronaut/brains/meta-intent-map.json');
  try {
    const raw = fs.readFileSync(intentPath, 'utf8');
    return JSON.parse(raw);
  } catch {
    console.warn('[Test] Could not load meta-intent-map.json, using inline fallback');
    return {
      intents: {
        control:     { target: 'CM-1', trigger_bigrams: ['phase boundary', 'scope gate', 'control signal'] },
        perceive:    { target: 'PM-1', trigger_bigrams: ['input field', 'noise filter', 'select field'] },
        schedule:    { target: 'TM-1', trigger_bigrams: ['collapse schedule', 'replay gate', 'phase align'] },
        detect_host: { target: 'HM-1', trigger_bigrams: ['host capability', 'io normalize', 'probe platform'] },
        store:       { target: 'SM-1', trigger_bigrams: ['store object', 'retrieve object', 'byte identity'] },
        infer:       { target: 'MM-1', trigger_bigrams: ['token signal', 'model voice', 'emit token'] },
        expand:      { target: 'XM-1', trigger_bigrams: ['expand narrative', 'generate metaphor', 'provide analogy'] },
        render:      { target: 'VM-1', trigger_bigrams: ['render projection', 'render svg', 'render css'] },
        verify:      { target: 'VM-2', trigger_bigrams: ['proof check', 'verify replay', 'attest hash'] },
      },
      routing: { fallback: 'XM-1', minimum_confidence: 0.3 },
    };
  }
}

// ---------------------------------------------------------------------------
// Run tests
// ---------------------------------------------------------------------------

async function runTests() {
  console.log('='.repeat(70));
  console.log('MICRONAUT INFERENCE CLUSTER — TEST RUNNER');
  console.log('='.repeat(70));
  console.log();

  // 1. Create the 1000-node cluster
  console.log('[1/5] Building inference cluster...');
  const cluster = createInferenceCluster(10);
  const clusterStatus = cluster.getStatus();
  console.log(`  Nodes: ${clusterStatus.totalNodes}`);
  console.log(`  Types: ${JSON.stringify(clusterStatus.typeDistribution)}`);
  console.log();

  // 2. Load intent map
  console.log('[2/5] Loading intent map...');
  const intentMap = loadIntentMap();
  console.log(`  Intents loaded: ${Object.keys(intentMap.intents).length}`);
  console.log();

  // 3. Initialize Phi-2 loader (stub mode — no actual model needed)
  console.log('[3/5] Initializing Phi-2 GGUF loader (stub mode)...');
  const phi2 = new Phi2GGUFLoader();
  const loadStatus = await phi2.load();
  console.log(`  Loaded: ${loadStatus.loaded}, Mode: ${loadStatus.mode || 'real'}`);
  console.log();

  // 4. Run test prompts through the pipeline
  console.log('[4/5] Running test prompts through MoE pipeline...');
  console.log();

  const testResults = [];
  let passed = 0;
  let failed = 0;

  for (const test of TEST_PROMPTS) {
    console.log(`  [${test.id}] "${test.prompt.slice(0, 60)}..."`);

    // Run through cluster pipeline
    const pipelineResult = await cluster.runInference(test.prompt, intentMap);

    // Run through Phi-2 loader (stub mode) for token generation
    phi2.openControlGate({
      decide_hash: hashData(`gate_${test.id}`),
      policy_hash: hashData('test_policy'),
      target_fold: '⟁COMPUTE_FOLD⟁',
    });

    const tokenResult = await phi2.emitToken(test.prompt);
    const voiceResult = await phi2.voiceModel(test.prompt, { max_tokens: 8 });
    const logitResult = await phi2.scoreLogits(test.prompt);

    phi2.closeControlGate();

    // Check expert routing
    const routedCorrectly = pipelineResult.selectedExpert === test.expectedExpert;
    if (routedCorrectly) {
      passed++;
      console.log(`    ✓ Routed to ${pipelineResult.selectedExpert} (expected ${test.expectedExpert})`);
    } else {
      failed++;
      console.log(`    ✗ Routed to ${pipelineResult.selectedExpert} (expected ${test.expectedExpert})`);
    }

    console.log(`    Stages: ${pipelineResult.stages}, Nodes: ${pipelineResult.totalNodesActivated}`);
    console.log(`    Token: "${tokenResult.token.slice(0, 20)}...", Voice: "${voiceResult.text.slice(0, 30)}..."`);
    console.log();

    testResults.push({
      test: test.id,
      category: test.category,
      prompt: test.prompt,
      expectedExpert: test.expectedExpert,
      actualExpert: pipelineResult.selectedExpert,
      routedCorrectly,
      pipeline: {
        id: pipelineResult.pipelineId,
        stages: pipelineResult.stages,
        totalNodesActivated: pipelineResult.totalNodesActivated,
        finalHash: pipelineResult.finalHash,
        deterministic: pipelineResult.deterministic,
        trace: pipelineResult.trace.map(stage => ({
          stage: stage.stage,
          micronaut: stage.micronaut,
          action: stage.action,
          nodesUsed: stage.nodesUsed,
          outputHash: stage.outputHash,
          phase: stage.phase || null,
          cm1_code: stage.cm1_code || null,
        })),
      },
      phi2_inference: {
        emit_token: {
          tool: tokenResult.tool,
          token: tokenResult.token,
          hash: tokenResult.hash,
          fold: tokenResult.fold,
          lane: tokenResult.lane,
        },
        voice_model: {
          tool: voiceResult.tool,
          text: voiceResult.text,
          tokens_generated: voiceResult.tokens_generated,
          hash: voiceResult.hash,
        },
        score_logits: {
          tool: logitResult.tool,
          candidates: logitResult.candidates,
          hash: logitResult.hash,
        },
      },
    });
  }

  // 5. Decay and get final cluster state
  console.log('[5/5] Collecting final cluster statistics...');
  cluster.tickDecay(0.01);

  const finalStatus = cluster.getStatus();
  const phi2Status = phi2.getStatus();

  // Build the output
  const output = {
    _schema: 'micronaut://test/inference-cluster/v1',
    _generated: new Date().toISOString(),
    _authority: 'KUHUL_π',
    _deterministic: true,

    summary: {
      total_tests: TEST_PROMPTS.length,
      passed,
      failed,
      pass_rate: parseFloat((passed / TEST_PROMPTS.length).toFixed(2)),
      total_pipeline_runs: finalStatus.pipelineRuns,
      total_node_inferences: finalStatus.totalInferences,
    },

    model: {
      name: phi2Status.model,
      format: phi2Status.format,
      quantization: phi2Status.quantization,
      stub_mode: phi2Status.stubMode,
      inference_count: phi2Status.inferenceCount,
      deterministic: phi2Status.deterministic,
      fold: phi2Status.fold,
      micronaut: phi2Status.micronaut,
      lane: phi2Status.lane,
    },

    cluster: {
      total_nodes: finalStatus.totalNodes,
      grid_size: finalStatus.gridSize,
      active_nodes: finalStatus.activeNodes,
      type_distribution: finalStatus.typeDistribution,
      fold_stats: finalStatus.foldStats,
    },

    moe_routing: {
      strategy: 'ngram_match_score',
      fallback: 'XM-1',
      intents_loaded: Object.keys(intentMap.intents).length,
      routing_accuracy: parseFloat((passed / TEST_PROMPTS.length).toFixed(2)),
    },

    inference_settings: {
      temperature: 0.0,
      top_k: 1,
      top_p: 1.0,
      repetition_penalty: 1.0,
      seed: 42,
      max_tokens: 512,
      deterministic: true,
      note: 'temperature=0.0 + top_k=1 = argmax (fully deterministic)',
    },

    tests: testResults,
  };

  // Write output
  const outputPath = path.resolve(__dirname, '../test_result.json');
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

  console.log();
  console.log('='.repeat(70));
  console.log(`RESULTS: ${passed}/${TEST_PROMPTS.length} passed (${(passed / TEST_PROMPTS.length * 100).toFixed(0)}%)`);
  console.log(`Output: ${outputPath}`);
  console.log('='.repeat(70));

  return output;
}

function hashData(data) {
  return crypto.createHash('sha256').update(String(data), 'utf8').digest('hex');
}

// Run
runTests().catch(err => {
  console.error('Test runner failed:', err);
  process.exit(1);
});
