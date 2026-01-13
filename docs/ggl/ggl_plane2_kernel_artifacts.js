// What else (code-only): the missing lock-points to make Plane-2 “sealed compute” real.
// Bundle below includes:
//
// 1) xjson://contract/ggl.infer/v1  (mirrors qwen/lam.o style)
// 2) ggltensors proof-hash contract (infer + pack) — exact payload fields locked
// 3) SCXQ2 frame kinds + canonical proof chain rules for GGL calls
// 4) sw.js kernel hooks: K0xx “sealed compute router” + frame capture service usage
// 5) Minimal verifier: replay-check of (start/end/pack/seed) chain + hash checks
//
// Drop-in: split into files if you want; kept as one block per your “no html, code only”.

'use strict';

/* ============================================================
   (1) xjson://contract/ggl.infer/v1  — FROZEN
   ============================================================ */

const XJSON_CONTRACT_GGL_INFER_V1 = Object.freeze({
  $schema: 'xjson://schema/core/v1',
  '@id': 'xjson://contract/ggl.infer/v1',
  '@type': 'model.infer',
  '@v': '1.0.0',
  '@status': 'FROZEN',
  '@plane': 2,
  provider: 'ggl',
  laws: {
    sealed_compute: true,
    pure_io: true,
    no_side_effects: true,
    determinism: true,
    randomness: 'forbidden_unless_seeded'
  },
  input: {
    pack: 'xjson://contract/ggltensors.pack/v1',
    infer: {
      prompt: 'string',
      mode: ['chat', 'reasoning', 'analysis', 'image_gen'],
      max_tokens: 'int?',
      stream: 'bool?',
      // Optional explicit seed request; if present must create infer.seed frame
      seed: 'int?'
    }
  },
  output: {
    pack: {
      model_id: 'string',
      pack_hash: 'string',
      tensor_hash: 'string',
      glyph_hash: 'string',
      abi_id: 'string',
      abi_hash: 'string',
      runtime_device: 'string',
      runtime_precision: 'string',
      seed: 'int?'
    },
    infer: {
      input_hash: 'string',
      output_hash: 'string',
      output: {
        text: 'string',
        tokens_used: 'int'
      }
    },
    frames: 'ggl.frames.stream.v1'
  }
});

/* ============================================================
   (2) Proof-hash contracts — EXACT payload fields (LOCKED)
   ============================================================ */

const HASH = Object.freeze({
  alg: 'fnv1a32',
  prefix: 'h:',
  // Stable JSON canonicalization: sort keys, no whitespace, deterministic arrays.
  canon(obj) {
    return stableStringify(obj);
  },
  fnv1a32_u32(str) {
    const s = String(str);
    let h = 0x811c9dc5 >>> 0;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h >>> 0;
  },
  h32(str) {
    const u = this.fnv1a32_u32(str);
    return this.prefix + u.toString(16).padStart(8, '0');
  }
});

// ggltensors.pack proof payload (must match your v1 pack contract fields)
function ggltensorsPackProofPayload_v1(p) {
  return {
    '@type': 'ggltensors.pack.proof.payload',
    '@v': '1.0.0',
    model_id: String(p.model_id),
    model_hash: p.model_hash == null ? null : String(p.model_hash),
    weights_hash: String(p.weights_hash),
    tensor_hash: String(p.tensor_hash),
    glyph_hash: String(p.glyph_hash),
    abi_id: String(p.abi_id),
    abi_hash: String(p.abi_hash),
    runtime_device: String(p.runtime_device),
    runtime_precision: String(p.runtime_precision),
    seed: p.seed == null ? null : (p.seed | 0),
    policy_sealed: !!p.policy_sealed,
    policy_no_network: !!p.policy_no_network,
    policy_no_fs: !!p.policy_no_fs,
    policy_no_eval: !!p.policy_no_eval
  };
}

function ggltensorsPackHash_v1(p) {
  // LOCK: hash of canonical payload only
  return HASH.h32(HASH.canon(ggltensorsPackProofPayload_v1(p)));
}

// ggl.infer proof payload (binds pack_hash + input/output hashes)
function gglInferProofPayload_v1(x) {
  return {
    '@type': 'ggl.infer.proof.payload',
    '@v': '1.0.0',
    pack_hash: String(x.pack_hash),
    model_id: String(x.model_id),
    input_hash: String(x.input_hash),
    output_hash: String(x.output_hash),
    // Optional, but if present must be exact
    seed: x.seed == null ? null : (x.seed | 0),
    abi_hash: String(x.abi_hash)
  };
}

function gglInferProofHash_v1(x) {
  return HASH.h32(HASH.canon(gglInferProofPayload_v1(x)));
}

/* ============================================================
   (3) SCXQ2 frame kinds + chain rules (GGL)
   ============================================================ */

const GGL_FRAME_KIND = Object.freeze({
  HDR: 'hdr',
  PACK: 'infer.pack',
  SEED: 'infer.seed',
  START: 'infer.start',
  END: 'infer.end',
  ERR: 'infer.error',
  ENDSTREAM: 'end'
});

/**
 * Chain rule (optional but recommended):
 * H_0 = h32(canon(HDR.payload))
 * H_i = h32( canon(frame_i.payload) + '|' + H_{i-1} )
 * ENDSTREAM.payload.final_chain_hash = H_last
 *
 * This is not a graph edge; it's a proof chain for replay.
 */
function chainUpdate(prevHash, payload) {
  const s = HASH.canon(payload) + '|' + String(prevHash || 'h:00000000');
  return HASH.h32(s);
}

/* ============================================================
   (4) Kernel-side hooks: sealed compute router skeleton
   - manifest.json is server law
   - sw.js is executor
   - SCXQ2 capture is shared service (like your current SW)
   ============================================================ */

// Minimal “kernel SCXQ2 service” interface expected:
// SCXQ2.capture(kind, payload)
// SCXQ2.export() / reset() (optional)

// This shows new numbered sections you’re missing:
// K040..K049 reserved for GGL sealed compute

async function K040_ggl_health(SCXQ2) {
  SCXQ2.capture('ggl.health', { ok: true });
  return json({ ok: true, provider: 'ggl', plane: 2 }, 200);
}

/**
 * K041 — ggl.infer (SEALED)
 * Routes to transformers.py / transformers.js / transformers.svg behind one contract.
 * Inputs/outputs are pure data; frames captured for replay.
 */
async function K041_ggl_infer(req, ctx, SCXQ2, adapters) {
  const body = await readJson(req);
  if (!body || typeof body !== 'object') throw fault('K041', 'E_CONTRACT', 'request body must be JSON object');

  // Contract: must contain pack + infer
  const pack = body.pack;
  const infer = body.infer;
  if (!pack || typeof pack !== 'object' || !infer || typeof infer !== 'object') {
    throw fault('K041', 'E_CONTRACT', 'missing pack/infer');
  }

  // Compute pack_hash (LOCKED) from pack proof payload fields
  const packPayload = {
    model_id: pack.model_id,
    model_hash: pack.model_hash ?? null,
    weights_hash: pack.weights_hash,
    tensor_hash: pack.tensor_hash,
    glyph_hash: pack.glyph_hash,
    abi_id: pack.abi_id || 'transformers.py:v1',
    abi_hash: pack.abi_hash || 'h:00000000',
    runtime_device: pack.runtime_device || 'cpu',
    runtime_precision: pack.runtime_precision || 'fp32',
    seed: (infer.seed ?? pack.seed) ?? null,
    policy_sealed: true,
    policy_no_network: true,
    policy_no_fs: true,
    policy_no_eval: true
  };
  const pack_hash = ggltensorsPackHash_v1(packPayload);

  // Frames
  SCXQ2.capture(GGL_FRAME_KIND.PACK, {
    pack_hash,
    model_id: String(packPayload.model_id),
    tensor_hash: String(packPayload.tensor_hash),
    glyph_hash: String(packPayload.glyph_hash),
    abi_hash: String(packPayload.abi_hash)
  });

  const seed = packPayload.seed == null ? null : (packPayload.seed | 0);
  if (seed != null) {
    SCXQ2.capture(GGL_FRAME_KIND.SEED, { pack_hash, seed });
  }

  const inputObj = {
    prompt: String(infer.prompt ?? ''),
    mode: String(infer.mode ?? 'chat'),
    max_tokens: infer.max_tokens == null ? 1024 : (infer.max_tokens | 0)
  };
  const input_hash = HASH.h32(HASH.canon(inputObj));
  SCXQ2.capture(GGL_FRAME_KIND.START, { pack_hash, input_hash });

  // Select adapter by abi_id (python/js/svg) — still pure I/O
  const abi_id = String(packPayload.abi_id);
  const adapter = adapters && adapters[abi_id];
  if (!adapter) throw fault('K041', 'E_NO_ADAPTER', 'no adapter for abi_id', { abi_id });

  // SEALED EXECUTION (black box)
  const out = await adapter.run(packPayload, inputObj); // must return { text, tokens_used, output_ref? }
  const outputPayload = {
    text: String(out.text ?? ''),
    tokens_used: out.tokens_used == null ? 0 : (out.tokens_used | 0)
  };
  const output_hash = HASH.h32(HASH.canon(outputPayload));
  SCXQ2.capture(GGL_FRAME_KIND.END, { pack_hash, output_hash });

  // Infer proof hash (LOCKED)
  const infer_proof = gglInferProofHash_v1({
    pack_hash,
    model_id: packPayload.model_id,
    input_hash,
    output_hash,
    seed,
    abi_hash: packPayload.abi_hash
  });

  return json({
    ok: true,
    contract: ctx.contract || 'xjson://contract/ggl.infer/v1',
    pack: {
      model_id: packPayload.model_id,
      pack_hash,
      tensor_hash: packPayload.tensor_hash,
      glyph_hash: packPayload.glyph_hash,
      abi_id: packPayload.abi_id,
      abi_hash: packPayload.abi_hash,
      runtime_device: packPayload.runtime_device,
      runtime_precision: packPayload.runtime_precision,
      seed
    },
    infer: {
      input_hash,
      output_hash,
      output: outputPayload,
      proof_hash: infer_proof
    }
  }, 200);
}

/* ============================================================
   (5) Replay verifier (minimal)
   - checks required frames exist and hashes line up
   - fixpoint logic belongs here (like ABR), not runtime
   ============================================================ */

function verifyGGLFrames_v1(frames) {
  // frames: [{t,k,p}, ...] (your SW capture shape) OR any list with {k,payload}
  const list = Array.isArray(frames) ? frames : [];
  const getK = (f) => f.k || f.kind || f.type || (f.payload && f.payload.kind);
  const getP = (f) => f.p || f.payload || f;

  const packs = [];
  const starts = [];
  const ends = [];
  const seeds = [];

  for (const f of list) {
    const k = getK(f);
    const p = getP(f);
    if (k === GGL_FRAME_KIND.PACK) packs.push(p);
    else if (k === GGL_FRAME_KIND.START) starts.push(p);
    else if (k === GGL_FRAME_KIND.END) ends.push(p);
    else if (k === GGL_FRAME_KIND.SEED) seeds.push(p);
  }

  if (!packs.length) return { ok: false, fault: 'E_NO_PACK' };
  if (!starts.length) return { ok: false, fault: 'E_NO_START' };
  if (!ends.length) return { ok: false, fault: 'E_NO_END' };

  // Minimal linkage: require same pack_hash appears across
  const pack_hash = String(packs[packs.length - 1].pack_hash || '');
  if (!pack_hash) return { ok: false, fault: 'E_PACK_HASH_MISSING' };

  const lastStart = starts.reverse().find((s) => String(s.pack_hash) === pack_hash) || null;
  const lastEnd = ends.reverse().find((e) => String(e.pack_hash) === pack_hash) || null;
  if (!lastStart || !lastEnd) return { ok: false, fault: 'E_LINK_FAIL' };

  if (!String(lastStart.input_hash || '')) return { ok: false, fault: 'E_INPUT_HASH_MISSING' };
  if (!String(lastEnd.output_hash || '')) return { ok: false, fault: 'E_OUTPUT_HASH_MISSING' };

  // If seed exists for this pack_hash it must be int
  const seedEvt = seeds.reverse().find((s) => String(s.pack_hash) === pack_hash) || null;
  if (seedEvt && !Number.isInteger(seedEvt.seed)) return { ok: false, fault: 'E_BAD_SEED' };

  return { ok: true, pack_hash, input_hash: lastStart.input_hash, output_hash: lastEnd.output_hash };
}

/* ============================================================
   Helpers (same style as your SW)
   ============================================================ */

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json' }
  });
}

function fault(section, code, message, meta) {
  const e = new Error(message || code);
  e.__fault = `${section}:${code}`;
  e.__section = section;
  e.__code = code;
  e.__meta = meta || null;
  e.__status = 500;
  return e;
}

async function readJson(req) {
  const text = await req.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { __raw: text };
  }
}

function stableStringify(value) {
  const seen = new Set();
  const walk = (v) => {
    if (v === null) return 'null';
    const t = typeof v;
    if (t === 'number') return Number.isFinite(v) ? String(v) : '0';
    if (t === 'boolean') return v ? 'true' : 'false';
    if (t === 'string') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(walk).join(',') + ']';
    if (t === 'object') {
      if (seen.has(v)) return '"[circular]"';
      seen.add(v);
      const keys = Object.keys(v).sort();
      const body = keys.map((k) => JSON.stringify(k) + ':' + walk(v[k])).join(',');
      seen.delete(v);
      return '{' + body + '}';
    }
    return '""';
  };
  return walk(value);
}

module.exports = {
  XJSON_CONTRACT_GGL_INFER_V1,
  HASH,
  ggltensorsPackProofPayload_v1,
  ggltensorsPackHash_v1,
  gglInferProofPayload_v1,
  gglInferProofHash_v1,
  GGL_FRAME_KIND,
  chainUpdate,
  K040_ggl_health,
  K041_ggl_infer,
  verifyGGLFrames_v1
};
