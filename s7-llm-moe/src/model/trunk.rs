/// Shared Trunk — 20M parameter base transformer.
///
/// Configuration (from config.rs):
///   vocab   = 24,576
///   hidden  = 768
///   layers  = 6
///   heads   = 12
///   head_dim= 64
///   ffn_dim = 3072  (4× expansion)
///   rope    = yes (RoPE-64)
///
/// All 4 experts share this trunk.  The trunk runs first; its output is
/// projected from hidden=768 → expert_hidden=512 before being passed to the
/// selected expert.
use super::embedding::Embedding;
use super::layer::TransformerLayer;
use super::linear::Linear;
use super::rope::{trunk_rope, RopeTable};
use crate::inference::kv_cache::KVCache;
use crate::model::config::{TRUNK_LAYERS, TRUNK_HIDDEN, EXPERT_HIDDEN};

pub struct SharedTrunk {
    pub embedding:  Embedding,
    pub layers:     Vec<TransformerLayer>,
    pub proj:       Linear,   // [TRUNK_HIDDEN, EXPERT_HIDDEN] projection
    pub rope:       RopeTable,
}

impl SharedTrunk {
    /// Forward pass for the shared trunk.
    ///
    /// token_id: single token to process at position `pos`
    /// kv:       mutable KV cache for the trunk layers
    ///
    /// Returns i8 vec of length EXPERT_HIDDEN (projected to expert input dim).
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        kv: &mut KVCache,
    ) -> Vec<i8> {
        debug_assert_eq!(self.layers.len(), TRUNK_LAYERS);

        // Token embedding lookup.
        let mut hidden = self.embedding.lookup(token_id);

        // Pass through trunk transformer layers.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let kv_layer = &mut kv.trunk[layer_idx];
            hidden = layer.forward(&hidden, pos, kv_layer, &self.rope);
        }

        // Project from TRUNK_HIDDEN → EXPERT_HIDDEN.
        self.proj.forward(&hidden)
    }
}
