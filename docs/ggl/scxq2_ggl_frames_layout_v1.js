// SCXQ2 binary lane layout for GGL frames (DICT / FIELD / LANE / EDGE mapping)
// FILE: scxq2_ggl_frames_layout_v1.js
// This is a SPEC+REFERENCE layout (no external deps). Deterministic ordering is law.
// Goal: encode frame stream as compact binary while remaining replay-verifiable.
//
// Concepts:
// - DICT: string table of recurring symbols (frame kinds, keys, model ids, etc.)
// - FIELD: typed columns (u32, f32, str_ref, bytes_ref)
// - LANE: logical record type (HDR, PACK, SEED, START, END, ERR, ENDSTREAM)
// - EDGE: optional linkage (prev_hash -> next_hash), NOT "graph edges"; just chain proof.
//
// NOTE: This matches your ABR frame stream principle, but for Plane-2 calls.

'use strict';

const SCXQ2_GGL = Object.freeze({
  v: '1.0.0',
  magic: 0x47474c31, // 'GGL1'
  endian: 'LE',
  hash: { alg: 'fnv1a32', prefix: 'h:' },

  // ----------------------------
  // LANE IDs (u8)
  // ----------------------------
  LANE: Object.freeze({
    HDR: 1,        // stream header
    PACK: 2,       // infer.pack
    SEED: 3,       // infer.seed (optional)
    START: 4,      // infer.start
    END: 5,        // infer.end
    ERR: 6,        // infer.error (optional)
    ENDSTREAM: 7   // terminator
  }),

  // ----------------------------
  // FIELD TYPES (u8)
  // ----------------------------
  FT: Object.freeze({
    U8: 1,
    U16: 2,
    U32: 3,
    I32: 4,
    F32: 5,
    STR: 6,        // u32 dict index
    BYTES: 7       // bytes blob (len + payload) OR external ref
  }),

  // ----------------------------
  // DICT CATEGORIES (u8)
  // ----------------------------
  DICT_KIND: Object.freeze({
    SYM: 1,        // keys, frame names
    MODEL: 2,      // model_id, weights_id
    ABI: 3,        // abi_id
    MIME: 4,       // mime strings
    URI: 5         // uri strings (idb://..., mem://...)
  }),

  // ----------------------------
  // RECORD SCHEMAS PER LANE
  // (field_name, field_type)
  // ----------------------------
  SCHEMA: Object.freeze({
    // HDR: identifies spec + policy hash, starts chain
    1: [
      ['t_ms', 'U32'],            // coarse timestamp (ms mod 2^32)
      ['spec_v', 'STR'],          // e.g. "ggl.frames.v1"
      ['policy_hash', 'STR'],     // policy contract hash
      ['dict_count', 'U32'],      // dict entries following
      ['dict_bytes', 'BYTES']     // packed dict section
    ],

    // PACK: binds model + tensor_hash + glyph_hash + abi_hash
    2: [
      ['t_ms', 'U32'],
      ['pack_hash', 'STR'],       // proof hash of pack payload (h:........)
      ['model_id', 'STR'],
      ['tensor_hash', 'STR'],
      ['glyph_hash', 'STR'],
      ['abi_hash', 'STR'],
      ['weights_hash', 'STR'],
      ['runtime_device', 'STR'],
      ['runtime_precision', 'STR']
    ],

    // SEED: explicit seed event (only if present)
    3: [
      ['t_ms', 'U32'],
      ['pack_hash', 'STR'],
      ['seed_i32', 'I32']
    ],

    // START: input hash (and optional input bytes reference)
    4: [
      ['t_ms', 'U32'],
      ['pack_hash', 'STR'],
      ['call_id', 'U32'],         // monotonic per stream
      ['input_hash', 'STR'],
      ['input_ref', 'STR'],       // uri or dict ref
      ['input_bytes', 'U32']      // size if known
    ],

    // END: output hash and linkage back to start
    5: [
      ['t_ms', 'U32'],
      ['pack_hash', 'STR'],
      ['call_id', 'U32'],
      ['start_hash', 'STR'],      // hash of START record payload (optional)
      ['output_hash', 'STR'],
      ['output_ref', 'STR'],
      ['output_bytes', 'U32'],
      ['tokens_used', 'U32']
    ],

    // ERR: error event, still binds to call_id
    6: [
      ['t_ms', 'U32'],
      ['pack_hash', 'STR'],
      ['call_id', 'U32'],
      ['start_hash', 'STR'],
      ['err_code', 'STR'],
      ['err_msg', 'STR']
    ],

    // ENDSTREAM: closes chain
    7: [
      ['t_ms', 'U32'],
      ['final_chain_hash', 'STR'] // hash chaining all records (optional but recommended)
    ]
  }),

  // ----------------------------
  // DICT PACKING (v1)
  // ----------------------------
  // dict_bytes is a blob:
  //   u32 entry_count
  //   repeat entry_count:
  //     u8 kind
  //     u16 byte_len
  //     bytes utf8
  //
  // Stable ordering: (kind asc, text asc). Index is position in this list.
  //
  // All STR fields store u32 dict_index (not raw strings).
  // Hashes like "h:0123abcd" should also be in DICT to compress.
  //
  // ----------------------------
  // EDGE / CHAINING (no graph)
  // ----------------------------
  // Optional: each record can be hashed as:
  //   H_i = fnv1a32( canon(record_payload_bytes) || H_{i-1} )
  // and ENDSTREAM.final_chain_hash stores H_last.
  //
  // This is not "edges", it's a proof chain.
});

module.exports = { SCXQ2_GGL };
