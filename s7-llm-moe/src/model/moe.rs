/// S7-LLM-MOE-300M — top-level model.
///
/// Execution law (one token):
///   h       = SharedTrunk(token_id, pos)           → ℝ^1024
///   route   = LearnedRouter(h)                     → expert_id ∈ {0..8}
///   h'      = Expert[expert_id](h, pos, kv)        → ℝ^1024
///   logits  = LM_Head(h')                          → ℝ^32768
///   next    = argmax(logits)                        (greedy, deterministic)
///
/// Router selection is deterministic (argmax, no sampling).
/// Proof record includes: expert_id, SHA-256(router_logits_i8).
///
/// Fold binding: ⟁COMPUTE_FOLD⟁
/// Lane:         BATCH (SCXQ2 id=5)
/// Micronaut:    MM-1 (token_signal_generator)
/// CM-1 gate:    U+0002 STX (@control.body.begin) must precede inference.
use super::trunk::SharedTrunk;
use super::expert::Expert;
use super::router::{LearnedRouter, RouterOutput};
use super::linear::Linear;
use crate::inference::kv_cache::KVCache;
use crate::inference::proof::ProofRecord;
use crate::model::config::{VOCAB_SIZE, NUM_EXPERTS};

pub struct S7LlmMoe {
    pub trunk:   SharedTrunk,
    pub experts: Vec<Expert>,    // len = NUM_EXPERTS = 9
    pub router:  LearnedRouter,
    pub lm_head: Linear,         // [EXPERT_HIDDEN, VOCAB_SIZE] — weight-tied
}

/// Full output from one token forward pass.
pub struct ForwardOutput {
    pub logits:     Vec<i8>,
    pub route:      RouterOutput,
    pub proof:      ProofRecord,
}

impl S7LlmMoe {
    /// Single-token forward pass.
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        kv: &mut KVCache,
    ) -> ForwardOutput {
        debug_assert_eq!(self.experts.len(), NUM_EXPERTS);

        // 1. Shared trunk — produces universal representation.
        let h_shared = self.trunk.forward(token_id, pos, kv);

        // 2. Learned router — deterministic expert selection.
        let route = self.router.forward(&h_shared);

        // 3. Expert forward (only one expert active per token).
        let expert_out = self.experts[route.expert_id]
            .forward(&h_shared, pos, kv);

        // 4. LM head → vocab logits.
        let logits = self.lm_head.forward(&expert_out);
        debug_assert_eq!(logits.len(), VOCAB_SIZE);

        // 5. Build proof record (for replay verification).
        let proof = ProofRecord::build(
            token_id,
            pos,
            route.expert_id,
            &route.logits_i8,
            &self.experts[route.expert_id],
        );

        ForwardOutput { logits, route, proof }
    }

    /// Construct from a parsed .s7l file.
    /// Weights are wired from the FIELD lane into each component.
    pub fn from_s7(_file: &crate::s7l::S7File) -> Self {
        unimplemented!(
            "Wire S7LlmMoe-300M from .s7l FIELD lane (id=2). \
             Seal weights with: cargo run --bin s7-pack-moe -- \
             --weights-dir model/weights/ --vocab model/vocab.json \
             --out model/moe-300m.s7l"
        )
    }
}
