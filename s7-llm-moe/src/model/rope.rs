/// Rotary Position Encoding (RoPE) — INT8 approximation.
///
/// RoPE rotates Q and K vectors at each attention head by a position-dependent
/// angle.  In full FP32 this is:
///   q' = q * cos(θ) − q_rot * sin(θ)
///   k' = k * cos(θ) − k_rot * sin(θ)
///
/// For INT8 inference we precompute a lookup table of cos/sin values scaled
/// to i8 range (×127) and apply them via integer multiply-shift.
///
/// Table dimensions: [MAX_CONTEXT, HEAD_DIM/2]  (one entry per pair of dims).
use crate::model::config::{MAX_CONTEXT, TRUNK_HEAD_DIM};

/// Precomputed RoPE table for a given head dimension.
/// cos_sin[pos][pair] = (cos_i8, sin_i8)
pub struct RopeTable {
    pub head_dim: usize,
    /// Flattened: [pos * (head_dim/2) + pair] → (cos_i8, sin_i8)
    table: Vec<(i8, i8)>,
    pub max_pos: usize,
}

impl RopeTable {
    /// Build a RoPE table for `head_dim` dimensions and `max_pos` positions.
    pub fn build(head_dim: usize, max_pos: usize) -> Self {
        let pairs = head_dim / 2;
        let mut table = Vec::with_capacity(max_pos * pairs);
        for pos in 0..max_pos {
            for pair in 0..pairs {
                // Standard RoPE frequency: θ_i = pos / 10000^(2i/d)
                let freq = (pos as f64)
                    / 10_000_f64.powf(2.0 * pair as f64 / head_dim as f64);
                let cos_val = (freq.cos() * 127.0).round().clamp(-128.0, 127.0) as i8;
                let sin_val = (freq.sin() * 127.0).round().clamp(-128.0, 127.0) as i8;
                table.push((cos_val, sin_val));
            }
        }
        RopeTable { head_dim, table, max_pos }
    }

    /// Apply RoPE rotation to a single head vector at position `pos`.
    /// `head` must have length == head_dim.  Rotated in-place.
    pub fn rotate(&self, head: &mut Vec<i8>, pos: usize) {
        let pairs = self.head_dim / 2;
        let base  = pos * pairs;
        for pair in 0..pairs {
            let (cos_i8, sin_i8) = self.table[base + pair];
            let x0 = head[pair * 2]     as i32;
            let x1 = head[pair * 2 + 1] as i32;
            // Rotated: x0' = x0*cos - x1*sin ; x1' = x0*sin + x1*cos
            let rx0 = ((x0 * cos_i8 as i32 - x1 * sin_i8 as i32) >> 7)
                .clamp(-128, 127) as i8;
            let rx1 = ((x0 * sin_i8 as i32 + x1 * cos_i8 as i32) >> 7)
                .clamp(-128, 127) as i8;
            head[pair * 2]     = rx0;
            head[pair * 2 + 1] = rx1;
        }
    }
}

/// Global trunk RoPE table (built once at model load time).
pub fn trunk_rope() -> RopeTable {
    RopeTable::build(TRUNK_HEAD_DIM, MAX_CONTEXT)
}

/// Expert RoPE table (head_dim = EXPERT_HEAD_DIM).
pub fn expert_rope(head_dim: usize) -> RopeTable {
    RopeTable::build(head_dim, MAX_CONTEXT)
}
