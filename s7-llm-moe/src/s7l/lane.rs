use sha2::{Digest, Sha256};

/// A single SCXQ2 lane extracted from a .s7l file.
pub struct Lane {
    pub id:      u8,
    pub payload: Vec<u8>,
    /// SHA-256 of payload as stored in the file (already verified on parse).
    pub hash:    [u8; 32],
}

/// Parse all lanes from the raw file bytes.
/// Each lane record:
///   u8   id
///   u32  payload_len (big-endian)
///   [u8] payload (payload_len bytes)
///   [u8; 32] expected SHA-256 of payload
pub fn parse_lanes(data: &[u8]) -> Result<Vec<Lane>, &'static str> {
    let mut lanes = Vec::new();
    let mut i = 40usize; // skip 40-byte header

    while i < data.len() {
        if i + 5 + 32 > data.len() {
            return Err("lane header truncated");
        }
        let id = data[i];
        i += 1;

        let len = u32::from_be_bytes(
            data[i..i + 4].try_into().unwrap()
        ) as usize;
        i += 4;

        if i + len + 32 > data.len() {
            return Err("lane payload truncated");
        }
        let payload = data[i..i + len].to_vec();
        i += len;

        let mut stored_hash = [0u8; 32];
        stored_hash.copy_from_slice(&data[i..i + 32]);
        i += 32;

        let computed: [u8; 32] = Sha256::digest(&payload).into();
        if computed != stored_hash {
            return Err("lane hash mismatch — payload is corrupt");
        }

        lanes.push(Lane { id, payload, hash: stored_hash });
    }

    Ok(lanes)
}

/// Serialise a single lane to bytes (used by the packer binary).
pub fn serialise_lane(id: u8, payload: &[u8]) -> Vec<u8> {
    let hash: [u8; 32] = Sha256::digest(payload).into();
    let len = payload.len() as u32;
    let mut out = Vec::with_capacity(1 + 4 + payload.len() + 32);
    out.push(id);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&hash);
    out
}
