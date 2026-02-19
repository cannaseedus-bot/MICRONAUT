/// Per-expert KV Cache for S7-LLM-MOE-140M.
///
/// Architecture:
///   KVCache {
///       trunk:   [TRUNK_LAYERS]  KVLayer   — shared by all experts
///       experts: [4][EXPERT_LAYERS] KVLayer — one slice per expert
///   }
///
/// At each token, only the trunk cache and the active expert's cache grow.
/// Inactive expert caches are untouched → constant per-token memory cost.
///
/// Cache memory per token (INT8, seq_len = 2048):
///   Trunk:   2048 × 6 layers × 2 (K+V) × TRUNK_HIDDEN  =  2048 × 6 × 2 × 768  = 18.9MB
///   Expert:  2048 × 8 layers × 2 (K+V) × EXPERT_HIDDEN =  2048 × 8 × 2 × 512  = 16.8MB
///   Total:   ~35.7MB for full context (acceptable for WebGPU)
use crate::model::config::{
    TRUNK_LAYERS, EXPERT_LAYERS, TRUNK_HIDDEN, EXPERT_HIDDEN, NUM_EXPERTS,
};

/// Key-Value store for a single transformer layer.
/// keys[t] and values[t] are the K and V vectors at sequence position t.
pub struct KVLayer {
    pub keys:   Vec<Vec<i8>>,
    pub values: Vec<Vec<i8>>,
}

impl KVLayer {
    pub fn new() -> Self {
        Self { keys: Vec::new(), values: Vec::new() }
    }

    /// Append K and V for the current position.
    pub fn push(&mut self, key: Vec<i8>, value: Vec<i8>) {
        self.keys.push(key);
        self.values.push(value);
    }

    /// Current sequence length (number of cached positions).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Default for KVLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// Full MoE KV cache: trunk layers + 4 expert layer slices.
pub struct KVCache {
    pub trunk:   Vec<KVLayer>,             // [TRUNK_LAYERS]
    pub experts: Vec<Vec<KVLayer>>,        // [NUM_EXPERTS][EXPERT_LAYERS]
}

impl KVCache {
    /// Initialise an empty KV cache (no tokens cached yet).
    pub fn new() -> Self {
        KVCache {
            trunk: (0..TRUNK_LAYERS).map(|_| KVLayer::new()).collect(),
            experts: (0..NUM_EXPERTS)
                .map(|_| (0..EXPERT_LAYERS).map(|_| KVLayer::new()).collect())
                .collect(),
        }
    }

    /// Reset all caches (for a new sequence).
    pub fn reset(&mut self) {
        for l in &mut self.trunk {
            l.keys.clear();
            l.values.clear();
        }
        for expert_kv in &mut self.experts {
            for l in expert_kv.iter_mut() {
                l.keys.clear();
                l.values.clear();
            }
        }
    }

    /// Memory footprint (bytes) of the current cache state.
    pub fn memory_bytes(&self) -> usize {
        let trunk_bytes: usize = self.trunk.iter()
            .map(|l| l.keys.iter().map(|k| k.len()).sum::<usize>()
                   + l.values.iter().map(|v| v.len()).sum::<usize>())
            .sum();

        let expert_bytes: usize = self.experts.iter().flat_map(|e| e.iter())
            .map(|l| l.keys.iter().map(|k| k.len()).sum::<usize>()
                   + l.values.iter().map(|v| v.len()).sum::<usize>())
            .sum();

        trunk_bytes + expert_bytes
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}
