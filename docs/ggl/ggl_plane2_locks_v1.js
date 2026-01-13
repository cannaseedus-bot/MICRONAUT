// More "what else" (code-only): the remaining invariants + contracts + lane layout
// that complete the sealed Plane-2 story end-to-end.
//
// Adds:
// A) xjson://contract/ggltensors.pack/v1  (FROZEN) + pack proof hash contract
// B) xjson://contract/ggl.frames.stream/v1 (frames are first-class contract)
// C) SCXQ2 DICT/FIELD/LANE/EDGE binary lane layout for GGL frame streams (GGL-LANE-1)
// D) Kernel policy contract: manifest.kuhul_server.v1 schema (routes/caps/policies/contracts)
// E) Post-collapse barrier rule for GGL plane (same concept as ABR but for compute calls)
// F) Adapter ABI for transformers.py/js/svg (black box interface) + anti-leak checks
// G) Deterministic error framing (infer.error) + proof pin option
//
// Everything below is "locks + schemas + layouts" so models stop drifting.

'use strict';

/* ============================================================
   A) xjson://contract/ggltensors.pack/v1 — FROZEN
   Formalizes the pack as a contract (inputs+outputs+proof fields)
   ============================================================ */

const XJSON_CONTRACT_GGLTENSORS_PACK_V1 = Object.freeze({
  $schema: 'xjson://schema/core/v1',
  '@id': 'xjson://contract/ggltensors.pack/v1',
  '@type': 'ggltensors.pack',
  '@v': '1.0.0',
  '@status': 'FROZEN',
  laws: {
    determinism: true,
    no_network: true,
    no_fs: true,
    no_eval: true,
    pack_is_data: true
  },
  input: {
    // "what weights are executed" (or pointer to them)
    model_id: 'string',
    model_hash: 'string?',
    weights_hash: 'string',  // hash of weights artifact
    // "what tensor representation is used"
    tensor_hash: 'string',   // hash of ggltensors content
    glyph_hash: 'string',    // hash of glyph grammar/codex used to interpret tensors
    // "what runtime executes it"
    abi_id: 'string',        // transformers.py:v1 | transformers.js:v1 | transformers.svg:v1
    abi_hash: 'string',      // hash of adapter ABI contract (not code hash)
    runtime_device: 'string',    // cpu|webgpu|wasm|svg
    runtime_precision: 'string', // fp32|fp16|int8|glyph
    seed: 'int?',                // optional; if present must be recorded in frames
    // sealed compute policy pins
    policy_sealed: 'bool',
    policy_no_network: 'bool',
    policy_no_fs: 'bool',
    policy_no_eval: 'bool'
  },
  output: {
    pack_hash: 'string', // hash of canonical pack proof payload (LOCKED)
    proof: {
      '@type': 'ggltensors.pack.proof.payload',
      '@v': '1.0.0',
      fields_locked: true
    }
  }
});

/* ============================================================
   Hash + Canon (same as prior)
   ============================================================ */

const HASH = Object.freeze({
  alg: 'fnv1a32',
  prefix: 'h:',
  canon(obj) { return stableStringify(obj); },
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

// LOCKED payload fields for pack hash (exact)
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
  return HASH.h32(HASH.canon(ggltensorsPackProofPayload_v1(p)));
}

/* ============================================================
   B) Frames are first-class: xjson://contract/ggl.frames.stream/v1
   ============================================================ */

const XJSON_CONTRACT_GGL_FRAMES_STREAM_V1 = Object.freeze({
  $schema: 'xjson://schema/core/v1',
  '@id': 'xjson://contract/ggl.frames.stream/v1',
  '@type': 'ggl.frames.stream',
  '@v': '1.0.0',
  '@status': 'FROZEN',
  laws: {
    append_only: true,
    deterministic_order: true,
    replay_verifiable: true,
    no_side_effects: true
  },
  frame: {
    // Every frame must have these:
    k: 'string', // kind
    t: 'int',    // timestamp or monotonic tick; verifier may ignore wall time
    p: 'object', // payload
    // Optional: proof chain hash (if used)
    ch: 'string?' // chain hash
  },
  kinds: [
    'hdr',
    'route.start',
    'route.end',
    'infer.pack',
    'infer.seed',
    'infer.start',
    'infer.end',
    'infer.error',
    'end'
  ],
  required_sequences: [
    // Minimal for an inference call:
    ['infer.pack', 'infer.start', 'infer.end'],
    // Optional seed must appear after pack and before start:
    ['infer.pack', 'infer.seed?', 'infer.start']
  ]
});

/* ============================================================
   C) SCXQ2 Binary Lane Layout for GGL Frames — GGL-LANE-1 (FROZEN)
   DICT / FIELD / LANE / EDGE mapping
   ============================================================ */

const SCXQ2_GGL_LANE_1 = Object.freeze({
  '@id': 'scxq2://lane/ggl.frames/v1',
  '@status': 'FROZEN',
  MAGIC: 'GGL1',
  VERSION: 1,

  // DICT: string table (kinds, keys, small strings)
  // FIELD: column schema for frames
  // LANE: typed lanes for frame stream
  // EDGE: optional adjacency; in GGL we keep edge-free; EDGE reserved for future
  DICT: {
    lane_id: 0,
    encoding: 'utf8',
    purpose: 'dedupe kinds/keys/strings',
    entries: 'varint_count + [varint_len + bytes]*'
  },

  FIELD: {
    lane_id: 1,
    purpose: 'frame schema (fixed)',
    // field indices are frozen:
    // 0:k_kind_id (u16 dict id)
    // 1:t_u32 (u32)
    // 2:p_blob_id (u32 dict-or-blob id depending mode)
    // 3:ch_u32 (u32) optional; 0 means none
    fields: [
      { i: 0, name: 'k', type: 'u16', dict: true },
      { i: 1, name: 't', type: 'u32', dict: false },
      { i: 2, name: 'p', type: 'u32', dict_or_blob: true },
      { i: 3, name: 'ch', type: 'u32', optional: true }
    ]
  },

  LANE: {
    lane_id: 2,
    purpose: 'frames stream',
    record: 'k(u16) | t(u32) | p(u32) | ch(u32)',
    ordering: 'append_only',
    compression: 'dict_first'
  },

  EDGE: {
    lane_id: 3,
    purpose: 'reserved (must be empty in v1)',
    invariant: 'edge_free'
  }
});

/* ============================================================
   D) manifest.kuhul_server.v1 schema (server law contract)
   Routes/caps/policies/contracts live in manifest.json
   ============================================================ */

const XJSON_SCHEMA_MANIFEST_KUHUL_SERVER_V1 = Object.freeze({
  $schema: 'xjson://schema/core/v1',
  '@id': 'xjson://schema/manifest.kuhul_server.v1',
  '@type': 'schema',
  '@v': '1.0.0',
  '@status': 'FROZEN',
  shape: {
    v: 'string',
    kind: 'static_server_contract',
    authority: 'manifest.json',
    routes: 'object',   // path -> {method, handler, cap, contract?}
    caps: 'object',     // cap_id -> cap descriptor
    contracts: 'object',// contract_id -> inline XJSON contract objects
    policies: 'object'  // pins/allowlists/denies
  },
  invariants: [
    'routes[*].handler matches /^K[0-9]{3}$/',
    'caps[*].kind in {local_model, addon_model, vfs, tool, language, service}',
    'contract ids are xjson://contract/*',
    'policies.proof.pin.hash is h:******** (fnv1a32)'
  ]
});

/* ============================================================
   E) Post-collapse barrier for Plane-2 (sealed compute gating)
   After a collapse (answer emitted), compute calls are locked unless:
   - SAFE domain permits, OR
   - a new "epoch" begins (explicit reset) OR
   - proof.pin matches (capability)
   ============================================================ */

const GGL_BARRIER = Object.freeze({
  '@id': 'ggl.barrier.law/v1',
  '@status': 'FROZEN',
  states: {
    OPEN: 0,
    LOCKED: 1
  },
  // When locked, only allow caps with policy 'safe_after_collapse'
  default_policy: 'deny_after_collapse'
});

function isAllowedAfterCollapse(capDesc, barrierState) {
  if (barrierState !== GGL_BARRIER.states.LOCKED) return true;
  return !!capDesc && capDesc.policy === 'safe_after_collapse';
}

/* ============================================================
   F) Adapter ABI contracts for sealed runtimes (transformers.*)
   Black box interface must be identical across py/js/svg
   ============================================================ */

const XJSON_CONTRACT_TRANSFORMERS_ADAPTER_ABI_V1 = Object.freeze({
  $schema: 'xjson://schema/core/v1',
  '@id': 'xjson://contract/transformers.adapter.abi/v1',
  '@type': 'adapter.abi',
  '@v': '1.0.0',
  '@status': 'FROZEN',
  laws: {
    pure_io: true,
    no_network: true,
    no_fs: true,
    no_eval: true,
    deterministic_if_seeded: true,
    side_effects: 'forbidden'
  },
  fn: {
    name: 'run',
    input: {
      pack: 'xjson://contract/ggltensors.pack/v1.input',
      infer: {
        prompt: 'string',
        mode: 'string',
        max_tokens: 'int'
      }
    },
    output: {
      text: 'string',
      tokens_used: 'int'
    }
  }
});

// Anti-leak checker: adapter outputs may not contain control-channel fields
function sealedOutputCheck(out) {
  // Ensure only allowed fields exist
  const o = out && typeof out === 'object' ? out : {};
  const keys = Object.keys(o);
  for (const k of keys) {
    if (k !== 'text' && k !== 'tokens_used') return false;
  }
  if (typeof o.text !== 'string') return false;
  if (!Number.isInteger(o.tokens_used)) return false;
  return true;
}

/* ============================================================
   G) Deterministic error framing (infer.error) + optional proof pin
   ============================================================ */

function inferErrorFrame(kind, err, meta) {
  const section = err?.__section || 'K041';
  const code = err?.__code || 'E_UNKNOWN';
  const msg = String(err?.message || code);
  const payload = {
    fault: `${section}:${code}`,
    message: msg,
    meta: meta || err?.__meta || null
  };
  return { k: kind || 'infer.error', t: 0, p: payload };
}

function proofPinOK(pinHash, expectedHash) {
  return String(pinHash || '') === String(expectedHash || '');
}

/* ============================================================
   "What else?" checklist as code: FROZEN missing pieces you should add next
   ============================================================ */

const WHAT_ELSE_LOCKS = Object.freeze({
  // 1) Contract registry law
  contract_registry: {
    '@id': 'xjson://law/contracts.registry/v1',
    requires: [
      'xjson://contract/ggl.infer/v1',
      'xjson://contract/ggltensors.pack/v1',
      'xjson://contract/ggl.frames.stream/v1',
      'xjson://contract/transformers.adapter.abi/v1'
    ],
    invariant: 'contracts are immutable unless MAJOR bump'
  },

  // 2) Kernel section numbering law (so SW errors map to K###)
  kernel_sections: {
    '@id': 'kuhul://law/sw.sections/v1',
    invariant: [
      'every handler is K###_*',
      'router is K100_route',
      'manifest loader is K010_manifest_server',
      'sealed compute handlers live in K040-K049'
    ]
  },

  // 3) SCXQ2 stream service law (shared by all routes)
  scxq2_stream_service: {
    '@id': 'scxq2://service/stream.capture/v1',
    invariants: [
      'capture is append-only',
      'export returns deterministic order',
      'binary layout conforms to scxq2://lane/ggl.frames/v1'
    ]
  },

  // 4) Verifier responsibility law
  verifier_law: {
    '@id': 'kuhul://law/verifier.first/v1',
    invariants: [
      'fixpoint checks belong to verifier, not runtime',
      'replay correctness derives only from frames+contracts',
      'runtime may be swapped without changing proofs'
    ]
  }
});

/* ============================================================
   Helpers
   ============================================================ */

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
  XJSON_CONTRACT_GGLTENSORS_PACK_V1,
  XJSON_CONTRACT_GGL_FRAMES_STREAM_V1,
  XJSON_SCHEMA_MANIFEST_KUHUL_SERVER_V1,
  XJSON_CONTRACT_TRANSFORMERS_ADAPTER_ABI_V1,
  SCXQ2_GGL_LANE_1,
  HASH,
  ggltensorsPackProofPayload_v1,
  ggltensorsPackHash_v1,
  sealedOutputCheck,
  isAllowedAfterCollapse,
  inferErrorFrame,
  proofPinOK,
  WHAT_ELSE_LOCKS
};
