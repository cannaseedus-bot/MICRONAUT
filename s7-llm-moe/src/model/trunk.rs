/// Shared Trunk — S7-LLM-MOE-300M
///
/// Configuration:
///   vocab    = 32,768
///   hidden   = 1024
///   layers   = 12
///   heads    = 16
///   head_dim = 64
///   ffn_dim  = 4096  (4× expansion)
///   rope     = yes (RoPE-64)
///
/// All 9 experts share this trunk.
/// Trunk and expert hidden dims are equal (1024) — no projection needed.
use super::embedding::Embedding;
use super::layer::TransformerLayer;
use super::rope::{trunk_rope, RopeTable};
use crate::inference::kv_cache::KVCache;
use crate::model::config::TRUNK_LAYERS;

pub struct SharedTrunk {
    pub embedding: Embedding,
    pub layers:    Vec<TransformerLayer>,
    pub rope:      RopeTable,
}

impl SharedTrunk {
    /// Forward pass.
    ///
    /// token_id: single token to process at position `pos`
    /// kv:       mutable KV cache for the trunk layers
    ///
    /// Returns i8 vec of length TRUNK_HIDDEN = EXPERT_HIDDEN = 1024.
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        kv: &mut KVCache,
    ) -> Vec<i8> {
        debug_assert_eq!(self.layers.len(), TRUNK_LAYERS);

        let mut hidden = self.embedding.lookup(token_id);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let kv_layer = &mut kv.trunk[layer_idx];
            hidden = layer.forward(&hidden, pos, kv_layer, &self.rope);
        }

        // No projection: TRUNK_HIDDEN == EXPERT_HIDDEN in 300M design.
        hidden
    }
}
