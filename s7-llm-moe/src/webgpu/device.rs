/// WebGPU device initialisation for S7-LLM-MOE-140M.
///
/// Abstracts the wgpu device/queue pair and provides helper methods
/// for buffer allocation and shader dispatch.
///
/// All buffer uploads are deterministic (same weight bytes → same GPU buffer
/// hash), satisfying V6.
///
/// NOTE: This module is a design-complete stub.  Activate by adding
///       `wgpu = "0.19"` to Cargo.toml and enabling `--features webgpu`.
///       The interface contract is fully specified here.

/// Capabilities required from the WebGPU device.
/// These map to wgpu::Features flags.
pub struct DeviceRequirements {
    /// Minimum VRAM for model + KV cache + activations.
    pub min_vram_bytes: u64,
    /// Timestamp queries (used for deterministic timing proofs).
    pub timestamp_queries: bool,
    /// Shader f32 operations (always available in WebGPU).
    pub shader_f32: bool,
}

impl DeviceRequirements {
    pub fn for_moe_140m() -> Self {
        Self {
            min_vram_bytes:   200 * 1024 * 1024, // 200MB
            timestamp_queries: false,              // not required, optional
            shader_f32:        true,
        }
    }
}

/// Buffer types used in the inference pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    /// Read-only weight tensor (INT8 packed as i32).
    WeightReadOnly,
    /// Read-write activation tensor (f32).
    ActivationReadWrite,
    /// Uniform (small constants: dims, scales, route).
    Uniform,
    /// KV cache (read-write, grows per token).
    KVCache,
}

/// Descriptor for a GPU buffer allocation.
pub struct BufferDesc {
    pub name:       String,
    pub kind:       BufferKind,
    pub size_bytes: u64,
}

/// Pipeline dispatch configuration for a single compute pass.
pub struct DispatchConfig {
    pub shader_path:    &'static str,
    pub workgroup_x:    u32,
    pub workgroup_y:    u32,
    pub workgroup_z:    u32,
    pub bind_group_idx: u32,
}

/// Full inference pass descriptor for one token.
///
/// The WebGPU runtime executes this sequence of dispatches:
///   1. embedding_dispatch   → produces hidden[768] f32
///   2. trunk_layer_dispatch × TRUNK_LAYERS (attn + ffn each)
///   3. proj_dispatch        → hidden[768] → hidden[512]
///   4. router_dispatch      → selects expert (identity copy)
///   5. expert_layer_dispatch × EXPERT_LAYERS (for active expert only)
///   6. lm_head_dispatch     → logits[24576]
///   7. argmax_readback      → next_token u32 (CPU readback)
pub struct InferencePassDesc {
    pub token_id:  u32,
    pub pos:       usize,
    pub expert_id: u32,
}

/// Stub: returns the list of GPU buffer allocations needed for one
/// S7-LLM-MOE-140M inference pass.
///
/// When wgpu is added as a dependency, replace these stubs with actual
/// `wgpu::Device::create_buffer()` calls.
pub fn describe_buffers() -> Vec<BufferDesc> {
    use crate::model::config::*;
    let trunk_h  = TRUNK_HIDDEN as u64;
    let expert_h = EXPERT_HIDDEN as u64;
    let vocab    = VOCAB_SIZE as u64;
    let ctx      = MAX_CONTEXT as u64;

    let f32_bytes: u64 = 4;
    let i32_bytes: u64 = 4; // INT8 packed 4:1

    vec![
        // Embedding table (INT8 packed).
        BufferDesc {
            name:       "embedding.weight".into(),
            kind:       BufferKind::WeightReadOnly,
            size_bytes: (vocab * trunk_h / 4) * i32_bytes,
        },
        // Trunk KV cache (f32, all positions).
        BufferDesc {
            name:       "trunk.kv_cache".into(),
            kind:       BufferKind::KVCache,
            size_bytes: ctx * TRUNK_LAYERS as u64 * 2 * trunk_h * f32_bytes,
        },
        // Active expert KV cache (f32).
        BufferDesc {
            name:       "expert.kv_cache".into(),
            kind:       BufferKind::KVCache,
            size_bytes: ctx * EXPERT_LAYERS as u64 * 2 * expert_h * f32_bytes,
        },
        // Activation double-buffer (f32, reused per layer).
        BufferDesc {
            name:       "activation_a".into(),
            kind:       BufferKind::ActivationReadWrite,
            size_bytes: trunk_h.max(expert_h) * f32_bytes,
        },
        BufferDesc {
            name:       "activation_b".into(),
            kind:       BufferKind::ActivationReadWrite,
            size_bytes: trunk_h.max(expert_h) * f32_bytes,
        },
        // LM head logits (f32).
        BufferDesc {
            name:       "logits".into(),
            kind:       BufferKind::ActivationReadWrite,
            size_bytes: vocab * f32_bytes,
        },
    ]
}
