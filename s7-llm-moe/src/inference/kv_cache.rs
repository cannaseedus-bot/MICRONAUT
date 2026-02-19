/// Per-expert KV Cache for S7-LLM-MOE-300M.
///
/// Architecture:
///   KVCache {
///       trunk:   [TRUNK_LAYERS=12]  KVLayer   — shared by all 9 experts
///       experts: [9][EXPERT_LAYERS=4] KVLayer — one slice per expert
///   }
///
/// At each token:
///   - Trunk cache (12 layers) always grows.
///   - Only the selected expert's cache (4 layers) grows.
///   - The other 8 expert caches are untouched.
/// → Constant per-token memory cost.
///
/// Cache memory at max context (2048 tokens, INT8):
///   Trunk:   2048 × 12L × 2(K+V) × 1024 dim  = 50.3MB
///   Expert:  2048 × 4L  × 2(K+V) × 1024 dim  = 16.8MB  (one expert)
///   Total:   ~67MB at full context (all-f32: 134MB; INT8: 67MB)
use crate::model::config::{TRUNK_LAYERS, EXPERT_LAYERS, NUM_EXPERTS};

/// Key-Value store for a single transformer layer.
pub struct KVLayer {
    pub keys:   Vec<Vec<i8>>,
    pub values: Vec<Vec<i8>>,
}

impl KVLayer {
    pub fn new() -> Self {
        Self { keys: Vec::new(), values: Vec::new() }
    }

    pub fn push(&mut self, key: Vec<i8>, value: Vec<i8>) {
        self.keys.push(key);
        self.values.push(value);
    }

    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

impl Default for KVLayer {
    fn default() -> Self { Self::new() }
}

/// Full MoE KV cache: trunk layers + 9 expert layer slices.
pub struct KVCache {
    pub trunk:   Vec<KVLayer>,          // [TRUNK_LAYERS=12]
    pub experts: Vec<Vec<KVLayer>>,     // [NUM_EXPERTS=9][EXPERT_LAYERS=4]
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            trunk: (0..TRUNK_LAYERS).map(|_| KVLayer::new()).collect(),
            experts: (0..NUM_EXPERTS)
                .map(|_| (0..EXPERT_LAYERS).map(|_| KVLayer::new()).collect())
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.trunk {
            l.keys.clear(); l.values.clear();
        }
        for expert_kv in &mut self.experts {
            for l in expert_kv.iter_mut() {
                l.keys.clear(); l.values.clear();
            }
        }
    }

    /// Memory used by the cache in bytes.
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

    /// Expert-specific cache fill level (for load-balancing diagnostics).
    pub fn expert_cache_lengths(&self) -> Vec<usize> {
        self.experts.iter()
            .map(|layers| layers.iter().map(|l| l.len()).max().unwrap_or(0))
            .collect()
    }
}

impl Default for KVCache {
    fn default() -> Self { Self::new() }
}
