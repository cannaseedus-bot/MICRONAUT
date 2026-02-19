/// WebGPU compute pipeline definitions for S7-LLM-MOE-140M.
///
/// Each pipeline corresponds to one WGSL shader in `shaders/`.
/// Pipelines are created once at model load time and reused per token.
///
/// Workgroup sizing rationale:
///   - All shaders use workgroup_size(64) — maps to AMD wave64 and NVIDIA warp×2.
///   - For hidden=768: dispatch ceil(768/64) = 12 workgroups.
///   - For hidden=512: dispatch ceil(512/64) = 8 workgroups.
///   - For vocab=24576: dispatch ceil(24576/64) = 384 workgroups.

/// Pipeline IDs in dispatch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineId {
    EmbeddingLookup,
    MatmulInt8,       // Used for Q, K, V, O projections and FFN W1
    FfnGelu,          // FFN GELU + W2 projection
    Attention,        // Causal attention with KV cache
    RouterCopy,       // Deterministic router (identity copy)
    LmHead,           // Final vocab projection (reuses MatmulInt8)
}

/// Workgroup dispatch size for a given output dimension.
pub fn dispatch_x(output_dim: usize) -> u32 {
    ((output_dim + 63) / 64) as u32
}

/// Full pipeline execution plan for one token forward pass.
/// Returned as a list of (PipelineId, workgroup_x) pairs in order.
pub fn token_forward_plan(
    trunk_hidden:  usize,
    expert_hidden: usize,
    ffn_dim:       usize,
    vocab_size:    usize,
    trunk_layers:  usize,
    expert_layers: usize,
) -> Vec<(PipelineId, u32)> {
    let mut plan = Vec::new();

    // 1. Embedding lookup.
    plan.push((PipelineId::EmbeddingLookup, dispatch_x(trunk_hidden)));

    // 2. Trunk layers (attn + ffn per layer).
    for _ in 0..trunk_layers {
        // Q, K, V projections.
        plan.push((PipelineId::MatmulInt8, dispatch_x(trunk_hidden))); // Q
        plan.push((PipelineId::MatmulInt8, dispatch_x(trunk_hidden))); // K
        plan.push((PipelineId::MatmulInt8, dispatch_x(trunk_hidden))); // V
        // Attention.
        plan.push((PipelineId::Attention,  dispatch_x(trunk_hidden)));
        // O projection.
        plan.push((PipelineId::MatmulInt8, dispatch_x(trunk_hidden))); // O
        // FFN W1.
        plan.push((PipelineId::MatmulInt8, dispatch_x(ffn_dim)));
        // FFN GELU + W2.
        plan.push((PipelineId::FfnGelu,    dispatch_x(trunk_hidden)));
    }

    // 3. Trunk → expert projection.
    plan.push((PipelineId::MatmulInt8, dispatch_x(expert_hidden)));

    // 4. Router copy (selects active expert buffer).
    plan.push((PipelineId::RouterCopy, dispatch_x(expert_hidden)));

    // 5. Expert layers.
    let expert_ffn_dim = expert_hidden * 4;
    for _ in 0..expert_layers {
        plan.push((PipelineId::MatmulInt8, dispatch_x(expert_hidden))); // Q
        plan.push((PipelineId::MatmulInt8, dispatch_x(expert_hidden))); // K
        plan.push((PipelineId::MatmulInt8, dispatch_x(expert_hidden))); // V
        plan.push((PipelineId::Attention,  dispatch_x(expert_hidden)));
        plan.push((PipelineId::MatmulInt8, dispatch_x(expert_hidden))); // O
        plan.push((PipelineId::MatmulInt8, dispatch_x(expert_ffn_dim)));
        plan.push((PipelineId::FfnGelu,    dispatch_x(expert_hidden)));
    }

    // 6. LM head.
    plan.push((PipelineId::LmHead, dispatch_x(vocab_size)));

    plan
}

/// Compute total GPU dispatch count for one token.
pub fn dispatch_count_per_token(
    trunk_layers: usize,
    expert_layers: usize,
) -> usize {
    1                           // embedding
    + trunk_layers  * 7         // Q, K, V, attn, O, W1, GELU+W2
    + 1                         // trunk→expert proj
    + 1                         // router copy
    + expert_layers * 7         // expert Q,K,V,attn,O,W1,GELU+W2
    + 1                         // LM head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::*;

    #[test]
    fn plan_dispatch_count() {
        let plan = token_forward_plan(
            TRUNK_HIDDEN, EXPERT_HIDDEN, TRUNK_FFN_DIM,
            VOCAB_SIZE, TRUNK_LAYERS, EXPERT_LAYERS,
        );
        let expected = dispatch_count_per_token(TRUNK_LAYERS, EXPERT_LAYERS);
        assert_eq!(plan.len(), expected,
            "plan has {} dispatches, expected {}", plan.len(), expected);
    }
}
