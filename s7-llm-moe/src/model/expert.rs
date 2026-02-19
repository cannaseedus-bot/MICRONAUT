/// Domain Expert — one of four 30M-parameter transformers.
///
/// Configuration per expert (from config.rs):
///   hidden   = 512
///   layers   = 8
///   heads    = 8
///   head_dim = 64
///   ffn_dim  = 2048  (4× expansion)
///
/// Experts are domain-specialized:
///   Expert 0: Code
///   Expert 1: Math
///   Expert 2: Reasoning
///   Expert 3: General instruction
///
/// Input: EXPERT_HIDDEN-dim vector from SharedTrunk projection.
/// Output: EXPERT_HIDDEN-dim vector → LM head.
use super::layer::TransformerLayer;
use super::rope::{expert_rope, RopeTable};
use crate::inference::kv_cache::KVCache;
use crate::model::config::{EXPERT_LAYERS, EXPERT_HIDDEN, EXPERT_HEAD_DIM};

pub struct Expert {
    pub domain_id: usize,             // 0=Code, 1=Math, 2=Reason, 3=General
    pub layers:    Vec<TransformerLayer>,
    pub rope:      RopeTable,
}

impl Expert {
    /// Forward pass for one expert.
    ///
    /// trunk_out: i8 vec of length EXPERT_HIDDEN (output of trunk projection)
    /// pos:       current sequence position
    /// kv:        mutable KV cache (only this expert's cache slice is used)
    ///
    /// Returns i8 vec of length EXPERT_HIDDEN (input to LM head).
    pub fn forward(
        &self,
        trunk_out: &[i8],
        pos: usize,
        kv: &mut KVCache,
    ) -> Vec<i8> {
        debug_assert_eq!(self.layers.len(), EXPERT_LAYERS);
        debug_assert_eq!(trunk_out.len(), EXPERT_HIDDEN);

        let mut hidden = trunk_out.to_vec();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let kv_layer = &mut kv.experts[self.domain_id][layer_idx];
            hidden = layer.forward(&hidden, pos, kv_layer, &self.rope);
        }
        hidden
    }
}

/// Build a zero-weight Expert with given domain_id (used during weight loading).
/// Actual weights are wired in by the S7 file deserialiser.
pub fn build_expert_placeholder(domain_id: usize) -> Expert {
    use super::linear::Linear;
    use super::attention::MultiHeadAttention;
    use super::ffn::FFN;
    use super::layer::TransformerLayer;
    use crate::tensor::Int8Tensor;
    use crate::model::config::{EXPERT_HEADS, EXPERT_FFN_DIM};

    let head_dim = EXPERT_HEAD_DIM;
    let hidden   = EXPERT_HIDDEN;
    let ffn_dim  = EXPERT_FFN_DIM;

    let make_linear = |name: &str, rows: usize, cols: usize| Linear {
        weight: Int8Tensor::zeros(name, vec![rows, cols], 1.0 / 127.0),
    };

    let layers = (0..EXPERT_LAYERS)
        .map(|l| TransformerLayer {
            attn: MultiHeadAttention {
                q_proj: make_linear(&format!("expert{}.layer{}.attn.q_proj.weight", domain_id, l), hidden, hidden),
                k_proj: make_linear(&format!("expert{}.layer{}.attn.k_proj.weight", domain_id, l), hidden, hidden),
                v_proj: make_linear(&format!("expert{}.layer{}.attn.v_proj.weight", domain_id, l), hidden, hidden),
                o_proj: make_linear(&format!("expert{}.layer{}.attn.o_proj.weight", domain_id, l), hidden, hidden),
                n_heads:  EXPERT_HEADS,
                head_dim,
            },
            ffn: FFN {
                fc1: make_linear(&format!("expert{}.layer{}.ffn.fc1.weight", domain_id, l), hidden, ffn_dim),
                fc2: make_linear(&format!("expert{}.layer{}.ffn.fc2.weight", domain_id, l), ffn_dim, hidden),
            },
        })
        .collect();

    Expert {
        domain_id,
        layers,
        rope: expert_rope(head_dim),
    }
}
