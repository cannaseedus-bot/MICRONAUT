/// Multi-head Self-Attention with RoPE and KV cache.
///
/// Operates entirely in INT8/INT32 arithmetic.
/// Softmax is approximated by integer argmax-normalised scaling
/// (exact softmax values are never computed — only the argmax token matters).
use super::linear::Linear;
use super::rope::RopeTable;
use crate::inference::kv_cache::KVLayer;

pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub n_heads:   usize,
    pub head_dim:  usize,
}

impl MultiHeadAttention {
    /// Single-token causal attention with KV cache.
    ///
    /// hidden_in: i8 vector of length hidden_dim
    /// pos:       current sequence position (0-indexed)
    /// kv:        mutable KV cache for this layer
    /// rope:      RoPE table (shared by all heads at same head_dim)
    ///
    /// Returns i8 vector of length hidden_dim.
    pub fn forward(
        &self,
        hidden_in: &[i8],
        pos: usize,
        kv: &mut KVLayer,
        rope: &RopeTable,
    ) -> Vec<i8> {
        let hidden_dim = self.n_heads * self.head_dim;

        // Project to Q, K, V
        let mut q = self.q_proj.forward(hidden_in); // [hidden_dim]
        let mut k = self.k_proj.forward(hidden_in); // [hidden_dim]
        let v     = self.v_proj.forward(hidden_in); // [hidden_dim]

        // Apply RoPE to Q and K (per-head rotation).
        for h in 0..self.n_heads {
            let start = h * self.head_dim;
            let end   = start + self.head_dim;
            let mut q_head = q[start..end].to_vec();
            let mut k_head = k[start..end].to_vec();
            rope.rotate(&mut q_head, pos);
            rope.rotate(&mut k_head, pos);
            q[start..end].copy_from_slice(&q_head);
            k[start..end].copy_from_slice(&k_head);
        }

        // Append K and V to cache.
        kv.push(k.clone(), v.clone());

        // Compute attention per head.
        let mut attended = vec![0i8; hidden_dim];

        for h in 0..self.n_heads {
            let h_start = h * self.head_dim;
            let h_end   = h_start + self.head_dim;
            let q_h     = &q[h_start..h_end];

            // Attention scores: q_h · k_cached[t]  for each t ≤ pos.
            let seq_len = kv.len();
            let mut scores: Vec<i32> = (0..seq_len)
                .map(|t| {
                    let k_h = &kv.keys[t][h_start..h_end];
                    q_h.iter()
                        .zip(k_h.iter())
                        .map(|(&qi, &ki)| (qi as i32) * (ki as i32))
                        .sum::<i32>()
                })
                .collect();

            // Scale by 1/√head_dim via bit-shift approximation.
            // √64 = 8 → right-shift by 3.
            let shift = (self.head_dim as f32).sqrt().log2().ceil() as u32;
            for s in &mut scores {
                *s >>= shift;
            }

            // Integer softmax: compute weighted sum of V vectors.
            // Strategy: find max score, shift down, treat as unnormalised weights.
            let max_score = scores.iter().copied().max().unwrap_or(0);

            // Compute weighted V accumulation.
            let mut ctx_h = vec![0i32; self.head_dim];
            let mut weight_sum = 0i32;
            for (t, &score) in scores.iter().enumerate() {
                let w = (score - max_score).max(-32); // clamp so exp doesn't underflow
                // Integer approximation: e^w ≈ max(0, 32 + w) (linear in shaded region)
                let w_approx = (32 + w).max(0) as i32;
                weight_sum += w_approx;
                let v_h = &kv.values[t][h_start..h_end];
                for (d, &vi) in v_h.iter().enumerate() {
                    ctx_h[d] += (vi as i32) * w_approx;
                }
            }

            // Normalise and convert to i8.
            if weight_sum > 0 {
                for (d, val) in ctx_h.iter().enumerate() {
                    attended[h_start + d] = ((val / weight_sum) as i32)
                        .clamp(-128, 127) as i8;
                }
            }
        }

        // Output projection.
        self.o_proj.forward(&attended)
    }
}
