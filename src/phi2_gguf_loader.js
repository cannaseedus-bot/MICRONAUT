/**
 * Phi-2 GGUF Loader — Fold-Enclosed Model for MICRONAUT MoE System
 *
 * Loads TheBloke/phi-2-GGUF (Q2_K) via @huggingface/transformers (transformers.js)
 * and binds it to MM-1 (ModelMicronaut) within ⟁COMPUTE_FOLD⟁.
 *
 * The model is static, deterministic, and fold-enclosed:
 *   - Temperature 0.0, top_k 1, seed 42 → same input = same output
 *   - All inference flows through MM-1 tools (emit_token, stream_tokens, etc.)
 *   - CM-1 must gate every inference request before MM-1 can execute
 *   - VM-2 attests every inference result before it leaves ⟁COMPUTE_FOLD⟁
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Configuration (mirrors micronaut/models.toml)
// ---------------------------------------------------------------------------

const MODEL_CONFIG = {
  name: 'phi-2-gguf',
  source: 'TheBloke/phi-2-GGUF',
  format: 'gguf',
  quantization: 'Q2_K',
  modelPath: path.resolve(__dirname, '../models/phi-2/phi-2.Q2_K.gguf'),
  tokenizerPath: path.resolve(__dirname, '../models/phi-2/tokenizer.json'),
  configPath: path.resolve(__dirname, '../models/phi-2/config.json'),
  contextLength: 2048,
  vocabSize: 51200,

  // Deterministic inference — no randomness
  defaultParams: {
    max_tokens: 512,
    temperature: 0.0,
    top_p: 1.0,
    top_k: 1,
    repetition_penalty: 1.0,
    seed: 42,
  },

  // Fold binding
  boundMicronaut: 'MM-1',
  boundFold: '⟁COMPUTE_FOLD⟁',
  boundLane: 'BATCH',
};

// ---------------------------------------------------------------------------
// Phi2GGUFLoader class
// ---------------------------------------------------------------------------

export class Phi2GGUFLoader {
  constructor(config = MODEL_CONFIG) {
    this.config = config;
    this.model = null;
    this.tokenizer = null;
    this.pipeline = null;
    this.loaded = false;
    this.inferenceCount = 0;

    // CM-1 gate state — must be opened before inference
    this.controlGateOpen = false;
    this.currentPhase = '@control.null'; // U+0000 NUL
  }

  /**
   * Load the Phi-2 GGUF model via transformers.js
   * Returns a status object with load details.
   */
  async load() {
    const status = {
      model: this.config.name,
      format: this.config.format,
      quantization: this.config.quantization,
      fold: this.config.boundFold,
      micronaut: this.config.boundMicronaut,
      loaded: false,
      error: null,
    };

    try {
      // Attempt to import transformers.js dynamically
      // Supports both @xenova/transformers (v2) and @huggingface/transformers (v3+)
      let transformers;
      try {
        transformers = await import('@huggingface/transformers');
      } catch {
        try {
          transformers = await import('@xenova/transformers');
        } catch {
          // Fallback: provide a stub loader for environments without transformers.js
          console.warn('[Phi2GGUFLoader] transformers.js not installed — using stub mode');
          return this._loadStub(status);
        }
      }

      const { AutoTokenizer, AutoModelForCausalLM, pipeline: createPipeline } = transformers;

      // Check if local model files exist
      const modelExists = fs.existsSync(this.config.modelPath);

      if (modelExists) {
        // Load from local GGUF file
        console.log(`[Phi2GGUFLoader] Loading local GGUF: ${this.config.modelPath}`);
        this.tokenizer = await AutoTokenizer.from_pretrained(
          path.dirname(this.config.modelPath)
        );
        this.model = await AutoModelForCausalLM.from_pretrained(
          path.dirname(this.config.modelPath),
          { model_file_name: path.basename(this.config.modelPath) }
        );
      } else {
        // Load from HuggingFace Hub (will download GGUF)
        console.log(`[Phi2GGUFLoader] Downloading from ${this.config.source}`);
        this.pipeline = await createPipeline('text-generation', this.config.source, {
          quantized: true,
        });
      }

      this.loaded = true;
      status.loaded = true;
      console.log(`[Phi2GGUFLoader] Model loaded: ${this.config.name} (${this.config.quantization})`);
    } catch (err) {
      status.error = err.message;
      console.error(`[Phi2GGUFLoader] Load failed: ${err.message}`);
    }

    return status;
  }

  /**
   * Stub loader — provides deterministic mock inference when transformers.js
   * is not available. Useful for testing fold routing without the full model.
   */
  _loadStub(status) {
    this.loaded = true;
    this._stubMode = true;
    status.loaded = true;
    status.mode = 'stub';
    status.note = 'Using deterministic stub — install @huggingface/transformers for real inference';
    console.log('[Phi2GGUFLoader] Stub mode active — deterministic mock inference');
    return status;
  }

  // -------------------------------------------------------------------------
  // CM-1 Control Gate Integration
  // -------------------------------------------------------------------------

  /**
   * Open the CM-1 control gate for inference.
   * Must be called before any inference request.
   * Phase: U+0001 SOH → @control.header.begin (@Pop)
   */
  openControlGate(gateRecord) {
    if (!gateRecord || !gateRecord.decide_hash || !gateRecord.policy_hash) {
      throw new Error('V2 violation: control gate requires decide_hash and policy_hash');
    }

    this.controlGateOpen = true;
    this.currentPhase = '@control.header.begin'; // U+0001 SOH — @Pop
    this._gateRecord = gateRecord;

    return {
      gate: 'open',
      phase: this.currentPhase,
      cm1_code: 'U+0001',
      micronaut: 'CM-1',
      target_fold: this.config.boundFold,
    };
  }

  /**
   * Close the CM-1 control gate after inference completes.
   * Phase: U+0004 EOT → @control.transmission.end (@Collapse)
   */
  closeControlGate() {
    this.controlGateOpen = false;
    this.currentPhase = '@control.transmission.end'; // U+0004 EOT — @Collapse
    this._gateRecord = null;

    return {
      gate: 'closed',
      phase: this.currentPhase,
      cm1_code: 'U+0004',
      micronaut: 'CM-1',
    };
  }

  // -------------------------------------------------------------------------
  // MM-1 Tool Implementations (5 tools from micronaut-profiles.json)
  // -------------------------------------------------------------------------

  /**
   * emit_token — Emit a single token signal from the model.
   * Tool: MM-1.emit_token
   * Ngram triggers: ["emit token", "token signal"]
   */
  async emitToken(prompt, params = {}) {
    this._enforceGate('emit_token');
    this.currentPhase = '@control.body.begin'; // U+0002 STX — @Wo

    const inferenceParams = { ...this.config.defaultParams, ...params, max_tokens: 1 };
    const result = await this._runInference(prompt, inferenceParams);

    return {
      tool: 'emit_token',
      micronaut: 'MM-1',
      fold: this.config.boundFold,
      lane: this.config.boundLane,
      token: result.text,
      hash: this._hashResult(result.text),
      phase: this.currentPhase,
    };
  }

  /**
   * stream_tokens — Stream a sequence of token signals from model inference.
   * Tool: MM-1.stream_tokens
   * Ngram triggers: ["stream tokens", "token stream"]
   */
  async *streamTokens(prompt, params = {}) {
    this._enforceGate('stream_tokens');
    this.currentPhase = '@control.body.begin'; // U+0002 STX — @Wo

    const inferenceParams = { ...this.config.defaultParams, ...params };
    const result = await this._runInference(prompt, inferenceParams);

    // Split into individual tokens and yield them as a stream
    const tokens = result.text.split(/\s+/);
    for (let i = 0; i < tokens.length; i++) {
      yield {
        tool: 'stream_tokens',
        micronaut: 'MM-1',
        fold: this.config.boundFold,
        lane: this.config.boundLane,
        token: tokens[i],
        index: i,
        total: tokens.length,
        hash: this._hashResult(tokens[i]),
        phase: this.currentPhase,
      };
    }

    this.currentPhase = '@control.body.end'; // U+0003 ETX — @Sek
  }

  /**
   * voice_model — Provide the model voice for a given inference context.
   * Tool: MM-1.voice_model
   * Ngram triggers: ["model voice", "voice model"]
   */
  async voiceModel(prompt, params = {}) {
    this._enforceGate('voice_model');
    this.currentPhase = '@control.body.begin'; // U+0002 STX — @Wo

    const inferenceParams = { ...this.config.defaultParams, ...params };
    const result = await this._runInference(prompt, inferenceParams);

    this.currentPhase = '@control.body.end'; // U+0003 ETX — @Sek

    return {
      tool: 'voice_model',
      micronaut: 'MM-1',
      fold: this.config.boundFold,
      lane: this.config.boundLane,
      text: result.text,
      tokens_generated: result.tokensGenerated,
      hash: this._hashResult(result.text),
      phase: this.currentPhase,
    };
  }

  /**
   * score_logits — Score token logits and return ranked candidates.
   * Tool: MM-1.score_logits
   * Ngram triggers: ["score logits", "logit score"]
   */
  async scoreLogits(prompt, params = {}) {
    this._enforceGate('score_logits');
    this.currentPhase = '@control.body.begin'; // U+0002 STX — @Wo

    // In stub mode, generate deterministic logit scores from the prompt
    if (this._stubMode) {
      const words = prompt.split(/\s+/).slice(-5);
      const candidates = words.map((word, i) => ({
        token: word,
        logit: parseFloat((1.0 - i * 0.15).toFixed(4)),
        rank: i + 1,
      }));
      return {
        tool: 'score_logits',
        micronaut: 'MM-1',
        fold: this.config.boundFold,
        lane: this.config.boundLane,
        candidates,
        hash: this._hashResult(JSON.stringify(candidates)),
        phase: this.currentPhase,
      };
    }

    // With real model: get next-token logits
    const result = await this._runInference(prompt, { ...params, max_tokens: 1 });
    return {
      tool: 'score_logits',
      micronaut: 'MM-1',
      fold: this.config.boundFold,
      lane: this.config.boundLane,
      candidates: [{ token: result.text, logit: 1.0, rank: 1 }],
      hash: this._hashResult(result.text),
      phase: this.currentPhase,
    };
  }

  /**
   * sample_distribution — Sample from the token probability distribution.
   * Tool: MM-1.sample_distribution
   * Ngram triggers: ["sample distribution", "token probability"]
   *
   * Note: With temperature=0.0 and top_k=1, this is deterministic (argmax).
   */
  async sampleDistribution(prompt, params = {}) {
    this._enforceGate('sample_distribution');
    this.currentPhase = '@control.body.begin'; // U+0002 STX — @Wo

    const inferenceParams = { ...this.config.defaultParams, ...params };
    const result = await this._runInference(prompt, inferenceParams);

    return {
      tool: 'sample_distribution',
      micronaut: 'MM-1',
      fold: this.config.boundFold,
      lane: this.config.boundLane,
      sampled_text: result.text,
      tokens_generated: result.tokensGenerated,
      deterministic: inferenceParams.temperature === 0.0,
      hash: this._hashResult(result.text),
      phase: this.currentPhase,
    };
  }

  // -------------------------------------------------------------------------
  // Internal inference engine
  // -------------------------------------------------------------------------

  async _runInference(prompt, params) {
    this.inferenceCount++;
    const startTime = Date.now();

    let text = '';
    let tokensGenerated = 0;

    if (this._stubMode) {
      // Deterministic stub: hash-based generation
      const seed = this._hashResult(prompt + JSON.stringify(params));
      const stubTokens = [];
      for (let i = 0; i < Math.min(params.max_tokens, 32); i++) {
        const tokenHash = this._hashResult(seed + String(i));
        stubTokens.push(tokenHash.slice(0, 6));
      }
      text = stubTokens.join(' ');
      tokensGenerated = stubTokens.length;
    } else if (this.pipeline) {
      // transformers.js pipeline mode
      const output = await this.pipeline(prompt, {
        max_new_tokens: params.max_tokens,
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repetition_penalty: params.repetition_penalty,
        do_sample: params.temperature > 0,
      });
      text = output[0].generated_text.slice(prompt.length).trim();
      tokensGenerated = text.split(/\s+/).length;
    } else if (this.model && this.tokenizer) {
      // Direct model mode
      const inputs = await this.tokenizer(prompt, { return_tensors: true });
      const outputs = await this.model.generate({
        ...inputs,
        max_new_tokens: params.max_tokens,
        temperature: params.temperature,
        top_k: params.top_k,
        do_sample: params.temperature > 0,
      });
      text = await this.tokenizer.decode(outputs[0], { skip_special_tokens: true });
      text = text.slice(prompt.length).trim();
      tokensGenerated = text.split(/\s+/).length;
    }

    const elapsed = Date.now() - startTime;

    return {
      text,
      tokensGenerated,
      elapsed,
      inferenceId: this.inferenceCount,
      model: this.config.name,
      deterministic: params.temperature === 0.0,
    };
  }

  /**
   * Enforce CM-1 control gate is open before inference.
   * V2 rule: All mutations require explicit control gate records.
   */
  _enforceGate(toolName) {
    if (!this.loaded) {
      throw new Error(`[MM-1.${toolName}] Model not loaded`);
    }
    if (!this.controlGateOpen) {
      throw new Error(
        `[MM-1.${toolName}] V2 violation: CM-1 control gate not open. ` +
        `Call openControlGate() with a valid gate record before inference.`
      );
    }
  }

  /**
   * Deterministic SHA-256 hash for proof binding (V6, V7).
   */
  _hashResult(data) {
    return crypto.createHash('sha256').update(data, 'utf8').digest('hex');
  }

  // -------------------------------------------------------------------------
  // Status / introspection
  // -------------------------------------------------------------------------

  getStatus() {
    return {
      model: this.config.name,
      format: this.config.format,
      quantization: this.config.quantization,
      loaded: this.loaded,
      stubMode: !!this._stubMode,
      inferenceCount: this.inferenceCount,
      controlGateOpen: this.controlGateOpen,
      currentPhase: this.currentPhase,
      fold: this.config.boundFold,
      micronaut: this.config.boundMicronaut,
      lane: this.config.boundLane,
      deterministic: this.config.defaultParams.temperature === 0.0,
    };
  }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

export async function createPhi2Loader(customConfig = {}) {
  const config = { ...MODEL_CONFIG, ...customConfig };
  const loader = new Phi2GGUFLoader(config);
  const status = await loader.load();
  return { loader, status };
}

export default Phi2GGUFLoader;
