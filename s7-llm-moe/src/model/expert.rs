/// Learned Expert — S7-LLM-MOE-300M
///
/// Each expert is a 4-layer transformer subnetwork, hidden=1024.
/// There are 9 experts, one per micronaut/fold domain.
///
/// Experts share the same architecture; specialization emerges from
/// training via the load-balanced routing objective.
///
/// Expert IDs → Micronaut mapping (from config.rs):
///   0=PM-1  1=CM-1  2=TM-1  3=HM-1  4=MM-1
///   5=XM-1  6=SM-1  7=VM-2  8=VM-1
use super::layer::TransformerLayer;
use super::rope::{expert_rope, RopeTable};
use crate::inference::kv_cache::KVCache;
use crate::model::config::{
    EXPERT_LAYERS, EXPERT_HIDDEN, EXPERT_HEAD_DIM, EXPERT_HEADS, EXPERT_FFN_DIM,
    EXPERT_NAMES, EXPERT_FOLDS, NUM_EXPERTS,
};

pub struct Expert {
    pub expert_id:   usize,
    pub micronaut:   &'static str,
    pub fold:        &'static str,
    pub layers:      Vec<TransformerLayer>,
    pub rope:        RopeTable,
}

impl Expert {
    /// Forward pass for one expert.
    ///
    /// trunk_out: i8 vec of length EXPERT_HIDDEN (trunk output, same dim)
    /// pos:       current sequence position
    /// kv:        mutable KV cache (only this expert's slice is touched)
    pub fn forward(
        &self,
        trunk_out: &[i8],
        pos: usize,
        kv: &mut KVCache,
    ) -> Vec<i8> {
        debug_assert_eq!(self.layers.len(), EXPERT_LAYERS);
        debug_assert_eq!(trunk_out.len(), EXPERT_HIDDEN,
            "expert {} received wrong input dim", self.expert_id);

        let mut hidden = trunk_out.to_vec();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let kv_layer = &mut kv.experts[self.expert_id][layer_idx];
            hidden = layer.forward(&hidden, pos, kv_layer, &self.rope);
        }
        hidden
    }
}

/// Build a zero-weight Expert placeholder for the given expert_id.
/// Real weights are wired in by the .s7l FIELD lane deserialiser.
pub fn build_expert_placeholder(expert_id: usize) -> Expert {
    use super::linear::Linear;
    use super::attention::MultiHeadAttention;
    use super::ffn::FFN;
    use super::layer::TransformerLayer;
    use crate::tensor::Int8Tensor;

    let hidden   = EXPERT_HIDDEN;
    let ffn_dim  = EXPERT_FFN_DIM;
    let head_dim = EXPERT_HEAD_DIM;

    let make_linear = |name: &str, rows: usize, cols: usize| Linear {
        weight: Int8Tensor::zeros(name, vec![rows, cols], 1.0 / 127.0),
    };

    let layers = (0..EXPERT_LAYERS)
        .map(|l| {
            let pfx = format!("expert{}.layer{}", expert_id, l);
            TransformerLayer {
                attn: MultiHeadAttention {
                    q_proj: make_linear(&format!("{}.attn.q_proj.weight", pfx), hidden, hidden),
                    k_proj: make_linear(&format!("{}.attn.k_proj.weight", pfx), hidden, hidden),
                    v_proj: make_linear(&format!("{}.attn.v_proj.weight", pfx), hidden, hidden),
                    o_proj: make_linear(&format!("{}.attn.o_proj.weight", pfx), hidden, hidden),
                    n_heads:  EXPERT_HEADS,
                    head_dim,
                },
                ffn: FFN {
                    fc1: make_linear(&format!("{}.ffn.fc1.weight", pfx), hidden, ffn_dim),
                    fc2: make_linear(&format!("{}.ffn.fc2.weight", pfx), ffn_dim, hidden),
                },
            }
        })
        .collect();

    Expert {
        expert_id,
        micronaut: EXPERT_NAMES[expert_id],
        fold:      EXPERT_FOLDS[expert_id],
        layers,
        rope: expert_rope(head_dim),
    }
}

/// Build all 9 expert placeholders.
pub fn build_all_experts() -> Vec<Expert> {
    (0..NUM_EXPERTS).map(build_expert_placeholder).collect()
}
