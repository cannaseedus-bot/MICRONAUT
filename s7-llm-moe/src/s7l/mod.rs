pub mod header;
pub mod lane;
pub mod merkle;

use header::Header;
use lane::{parse_lanes, Lane};

/// Class byte for MoE LLM variant.
pub const CLASS_MOE_LLM: u8 = 0x02;

/// Flag bits.
pub const FLAG_INT8: u8 = 0x01;
pub const FLAG_MOE: u8  = 0x04;

/// A parsed, Merkle-verified .s7l artifact for the S7-LLM-MOE-140M model.
///
/// Layout (bytes):
///   [0..4]   Magic "S7LM"
///   [4..6]   Version (big-endian u16) = 0x0002
///   [6]      Class = 0x02 (MOE_LLM)
///   [7]      Flags = 0x05 (INT8 | MOE)
///   [8..40]  Root Merkle hash (SHA-256 over all lane-payload hashes)
///   [40..]   Lanes (repeating):
///              u8   lane_id
///              u32  payload_len (big-endian)
///              [u8] payload
///              [u8; 32] SHA-256 hash of payload (verified on parse)
pub struct S7File {
    pub header: Header,
    pub lanes:  Vec<Lane>,
}

impl S7File {
    /// Parse and Merkle-verify a .s7l byte slice.
    pub fn parse(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 40 {
            return Err("s7l too short — header requires 40 bytes minimum");
        }
        if &data[0..4] != b"S7LM" {
            return Err("bad magic — expected S7LM");
        }
        let header = Header::parse(data);
        if header.class != CLASS_MOE_LLM {
            return Err("wrong class byte — expected 0x02 (MOE_LLM)");
        }
        let lanes = parse_lanes(data)?;
        if !merkle::verify_root(&lanes, &header.root_hash) {
            return Err("Merkle root mismatch — sealed artifact is corrupt or tampered");
        }
        Ok(S7File { header, lanes })
    }

    /// Look up a lane by its SCXQ2 lane id (1=DICT, 2=FIELD, 3=LANE, 4=EDGE, 5=BATCH).
    pub fn lane(&self, id: u8) -> Option<&Lane> {
        self.lanes.iter().find(|l| l.id == id)
    }

    /// Lane id constants (SCXQ2 mapping).
    pub const LANE_DICT:  u8 = 1; // BPE vocabulary
    pub const LANE_FIELD: u8 = 2; // INT8 weight tensors
    pub const LANE_LANE:  u8 = 3; // Generation stream
    pub const LANE_EDGE:  u8 = 4; // Routing table + CM-1 topology + sub-Merkle roots
    pub const LANE_BATCH: u8 = 5; // Ephemeral compute
}
