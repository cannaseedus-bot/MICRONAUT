/// s7-pack-moe — packs trained weight files into a sealed .s7l artifact.
///
/// Usage:
///   cargo run --bin s7-pack-moe -- \
///     --weights-dir model/weights/ \
///     --vocab       model/vocab.json \
///     --out         model/moe.s7l
///
/// Weights directory layout (produced by the training pipeline):
///   weights/
///     embedding.weight.bin       (INT8, [24576, 768], AVX2-padded)
///     trunk.layer{0..5}.*.bin
///     expert{0..3}.layer{0..7}.*.bin
///     lm_head.weight.bin         (INT8, [512, 24576], AVX2-padded)
///     scales.json                (tensor name → f32 scale)
///
/// Output .s7l lanes:
///   Lane 1 DICT  — BPE vocab
///   Lane 2 FIELD — INT8 weight tensors (deterministic name-sorted order)
///   Lane 3 LANE  — empty placeholder
///   Lane 4 EDGE  — sub-Merkle roots (trunk + 4 experts)
///   Lane 5 BATCH — empty placeholder
///
/// Merkle sealing:
///   1. Compute SHA-256 of each lane payload.
///   2. Compute Merkle root over all lane hashes.
///   3. Write root into .s7l header.

use std::env;
use std::fs;
use std::path::Path;
use sha2::{Digest, Sha256};

// Pull in crate modules.
use s7_llm_moe::s7l::{
    header::Header,
    lane::serialise_lane,
    merkle::{compute_merkle_root, SubRoots},
    CLASS_MOE_LLM, FLAG_INT8, FLAG_MOE,
};
use s7_llm_moe::tensor::field_lane::write_tensor_record;
use s7_llm_moe::tensor::Int8Tensor;

fn main() {
    let args: Vec<String> = env::args().collect();
    let weights_dir = flag_val(&args, "--weights-dir").expect("--weights-dir required");
    let vocab_path  = flag_val(&args, "--vocab").expect("--vocab required");
    let out_path    = flag_val(&args, "--out").expect("--out required");

    eprintln!("[s7-pack-moe] Loading vocab from {}", vocab_path);
    let dict_payload = build_dict_lane(vocab_path);

    eprintln!("[s7-pack-moe] Loading weights from {}", weights_dir);
    let (field_payload, sub_roots) = build_field_lane(weights_dir);

    // Empty placeholders for LANE and BATCH.
    let lane_payload:  Vec<u8> = b"S7-LLM-MOE-140M generation stream".to_vec();
    let batch_payload: Vec<u8> = b"S7-LLM-MOE-140M ephemeral compute".to_vec();

    // EDGE lane: sub-Merkle roots (160 bytes) + padding.
    let edge_payload = sub_roots.to_bytes();

    // Serialise lanes.
    let lane1 = serialise_lane(1, &dict_payload);
    let lane2 = serialise_lane(2, &field_payload);
    let lane3 = serialise_lane(3, &lane_payload);
    let lane4 = serialise_lane(4, &edge_payload);
    let lane5 = serialise_lane(5, &batch_payload);

    // Compute per-lane payload hashes for Merkle root.
    let lane_hashes: Vec<[u8; 32]> = [
        &dict_payload, &field_payload, &lane_payload, &edge_payload, &batch_payload,
    ]
    .iter()
    .map(|p| Sha256::digest(p).into())
    .collect();

    let root_hash = compute_merkle_root(&lane_hashes);

    // Build .s7l header.
    let header = Header {
        magic:     *b"S7LM",
        version:   0x0002,
        class:     CLASS_MOE_LLM,
        flags:     FLAG_INT8 | FLAG_MOE,
        root_hash,
    };

    // Assemble the artifact.
    let mut artifact = Vec::new();
    artifact.extend_from_slice(&header.to_bytes());
    artifact.extend_from_slice(&lane1);
    artifact.extend_from_slice(&lane2);
    artifact.extend_from_slice(&lane3);
    artifact.extend_from_slice(&lane4);
    artifact.extend_from_slice(&lane5);

    fs::write(out_path, &artifact).expect("cannot write output file");
    eprintln!(
        "[s7-pack-moe] Sealed artifact written to {} ({} bytes)",
        out_path,
        artifact.len()
    );
    eprintln!("[s7-pack-moe] Merkle root: {}", hex(&root_hash));
}

/// Build DICT lane payload from the BPE vocab JSON file.
/// Format: repeated (u16 str_len, [u8] str, u32 id) records.
fn build_dict_lane(vocab_path: &str) -> Vec<u8> {
    let raw = fs::read_to_string(vocab_path).expect("cannot read vocab");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("bad JSON vocab");
    let obj = parsed.as_object().expect("vocab must be object");

    // Sort by token id for deterministic ordering (V0).
    let mut entries: Vec<(&str, u32)> = obj
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_u64().unwrap() as u32))
        .collect();
    entries.sort_by_key(|&(_, id)| id);

    let mut buf = Vec::new();
    for (token_str, id) in entries {
        let bytes = token_str.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(bytes);
        buf.extend_from_slice(&id.to_be_bytes());
    }
    buf
}

/// Build FIELD lane payload and sub-Merkle roots from the weights directory.
/// Tensors are written in deterministic name-sorted order (V0).
fn build_field_lane(weights_dir: &str) -> (Vec<u8>, SubRoots) {
    // Load scales.json mapping tensor name → f32 scale.
    let scales_path = format!("{}/scales.json", weights_dir);
    let scales_raw  = fs::read_to_string(&scales_path)
        .unwrap_or_else(|_| "{}".to_string());
    let scales_json: serde_json::Value = serde_json::from_str(&scales_raw).unwrap();

    let get_scale = |name: &str| -> f32 {
        scales_json
            .get(name)
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0 / 127.0)
    };

    // Collect all .bin files and sort by name (V0 determinism).
    let mut bin_files: Vec<String> = fs::read_dir(weights_dir)
        .expect("cannot read weights dir")
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".bin") { Some(name) } else { None }
        })
        .collect();
    bin_files.sort();

    // Build per-component payload buffers for sub-Merkle roots.
    let mut trunk_buf   = Vec::new();
    let mut expert_bufs = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut field_buf   = Vec::new();

    for filename in &bin_files {
        let tensor_name = filename.trim_end_matches(".bin");
        let path = format!("{}/{}", weights_dir, filename);
        let raw  = fs::read(&path)
            .unwrap_or_else(|_| panic!("cannot read {}", path));

        // Infer dims from the filename convention (see README tensor layout).
        let dims = infer_dims(tensor_name);
        let scale = get_scale(tensor_name);
        let data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();

        let tensor = Int8Tensor {
            name:  tensor_name.to_string(),
            dims,
            scale,
            data,
        };

        // Route to trunk or expert sub-buffer for Merkle computation.
        let mut record_buf = Vec::new();
        write_tensor_record(&tensor, &mut record_buf);
        write_tensor_record(&tensor, &mut field_buf);

        if tensor_name.starts_with("trunk") || tensor_name == "embedding.weight" {
            trunk_buf.extend_from_slice(&record_buf);
        } else if tensor_name.starts_with("expert0") {
            expert_bufs[0].extend_from_slice(&record_buf);
        } else if tensor_name.starts_with("expert1") {
            expert_bufs[1].extend_from_slice(&record_buf);
        } else if tensor_name.starts_with("expert2") {
            expert_bufs[2].extend_from_slice(&record_buf);
        } else if tensor_name.starts_with("expert3") {
            expert_bufs[3].extend_from_slice(&record_buf);
        }
    }

    // Compute sub-Merkle roots.
    let trunk_root: [u8; 32] = Sha256::digest(&trunk_buf).into();
    let expert_roots: [[u8; 32]; 4] = std::array::from_fn(|i| {
        Sha256::digest(&expert_bufs[i]).into()
    });

    let sub_roots = SubRoots {
        trunk:   trunk_root,
        experts: expert_roots,
    };

    (field_buf, sub_roots)
}

/// Infer tensor dimensions from the canonical name.
/// This is a best-effort inference; the training pipeline should emit
/// an explicit dims.json alongside the weights.
fn infer_dims(name: &str) -> Vec<usize> {
    use s7_llm_moe::model::config::*;
    match name {
        "embedding.weight"  => vec![VOCAB_SIZE, TRUNK_HIDDEN],
        "lm_head.weight"    => vec![EXPERT_HIDDEN, VOCAB_SIZE],
        _ if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") => {
            if name.starts_with("trunk") {
                vec![TRUNK_HIDDEN, TRUNK_HIDDEN]
            } else {
                vec![EXPERT_HIDDEN, EXPERT_HIDDEN]
            }
        },
        _ if name.contains("o_proj") => {
            if name.starts_with("trunk") {
                vec![TRUNK_HIDDEN, TRUNK_HIDDEN]
            } else {
                vec![EXPERT_HIDDEN, EXPERT_HIDDEN]
            }
        },
        _ if name.contains("ffn.fc1") => {
            if name.starts_with("trunk") {
                vec![TRUNK_HIDDEN, TRUNK_FFN_DIM]
            } else {
                vec![EXPERT_HIDDEN, EXPERT_FFN_DIM]
            }
        },
        _ if name.contains("ffn.fc2") => {
            if name.starts_with("trunk") {
                vec![TRUNK_FFN_DIM, TRUNK_HIDDEN]
            } else {
                vec![EXPERT_FFN_DIM, EXPERT_HIDDEN]
            }
        },
        _ if name.contains("proj") && name.contains("trunk") => {
            vec![TRUNK_HIDDEN, EXPERT_HIDDEN]
        },
        _ => vec![1], // unknown — will be overridden by dims.json if available
    }
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(&args[i + 1]);
        }
    }
    None
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
