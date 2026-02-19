use sha2::{Digest, Sha256};
use super::lane::Lane;

/// Compute a binary Merkle root over an ordered slice of 32-byte hashes.
/// Odd nodes are paired with themselves (standard Bitcoin-style).
pub fn compute_merkle_root(hashes: &[[u8; 32]]) -> [u8; 32] {
    if hashes.is_empty() {
        return [0u8; 32];
    }
    if hashes.len() == 1 {
        return hashes[0];
    }
    let mut layer: Vec<[u8; 32]> = hashes.to_vec();
    while layer.len() > 1 {
        let mut next: Vec<[u8; 32]> = Vec::with_capacity((layer.len() + 1) / 2);
        let mut j = 0;
        while j < layer.len() {
            let left  = layer[j];
            let right = if j + 1 < layer.len() { layer[j + 1] } else { layer[j] };
            let mut h = Sha256::new();
            h.update(left);
            h.update(right);
            next.push(h.finalize().into());
            j += 2;
        }
        layer = next;
    }
    layer[0]
}

/// Verify that the Merkle root computed over all lane payload-hashes matches
/// the root stored in the .s7l header.
pub fn verify_root(lanes: &[Lane], expected: &[u8; 32]) -> bool {
    let hashes: Vec<[u8; 32]> = lanes.iter().map(|l| l.hash).collect();
    compute_merkle_root(&hashes) == *expected
}

/// Per-component sub-Merkle roots for the MoE-300M model.
/// Stored in the EDGE lane payload as 10 consecutive 32-byte fields:
///   [0..32]    trunk_root
///   [32..64]   expert0_root  (PM-1)
///   [64..96]   expert1_root  (CM-1)
///   [96..128]  expert2_root  (TM-1)
///   [128..160] expert3_root  (HM-1)
///   [160..192] expert4_root  (MM-1)
///   [192..224] expert5_root  (XM-1)
///   [224..256] expert6_root  (SM-1)
///   [256..288] expert7_root  (VM-2)
///   [288..320] expert8_root  (VM-1)
/// GlobalRoot = SHA256(trunk_root || e0 || e1 || ... || e8)
pub struct SubRoots {
    pub trunk:   [u8; 32],
    pub experts: [[u8; 32]; 9],
}

impl SubRoots {
    pub fn global_root(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(self.trunk);
        for e in &self.experts {
            h.update(e);
        }
        h.finalize().into()
    }

    pub fn from_edge_lane(payload: &[u8]) -> Option<Self> {
        // 1 trunk + 9 experts = 10 × 32 = 320 bytes
        if payload.len() < 320 {
            return None;
        }
        let mut trunk = [0u8; 32];
        trunk.copy_from_slice(&payload[0..32]);
        let mut experts = [[0u8; 32]; 9];
        for (i, e) in experts.iter_mut().enumerate() {
            e.copy_from_slice(&payload[32 + i * 32..32 + (i + 1) * 32]);
        }
        Some(SubRoots { trunk, experts })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(320);
        out.extend_from_slice(&self.trunk);
        for e in &self.experts {
            out.extend_from_slice(e);
        }
        out
    }
}
