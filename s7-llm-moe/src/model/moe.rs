/// S7-LLM-MOE-140M — top-level model.
///
/// Execution law (one token):
///   x = SharedTrunk(token_id, pos)          → ℝ^EXPERT_HIDDEN
///   expert_id = DeterministicRouter(decoded) → {0,1,2,3}
///   x = Expert[expert_id](x, pos)           → ℝ^EXPERT_HIDDEN
///   logits = LM_Head(x)                     → ℝ^VOCAB_SIZE
///   next_token = argmax(logits)             (greedy, deterministic)
///
/// Fold binding: ⟁COMPUTE_FOLD⟁
/// Lane:         BATCH (SCXQ2 id=5)
/// Micronaut:    MM-1 (token_signal_generator)
/// CM-1 gate:    U+0002 STX (@control.body.begin) must precede inference.
use super::trunk::SharedTrunk;
use super::expert::Expert;
use super::router::{DeterministicRouter, ExpertDomain};
use super::linear::Linear;
use crate::inference::kv_cache::KVCache;
use crate::model::config::{VOCAB_SIZE, EXPERT_HIDDEN};

pub struct S7LlmMoe {
    pub trunk:   SharedTrunk,
    pub experts: Vec<Expert>,     // len = 4; indexed by ExpertDomain
    pub lm_head: Linear,          // [EXPERT_HIDDEN, VOCAB_SIZE] — weight-tied with embedding
    pub router:  DeterministicRouter,
}

impl S7LlmMoe {
    /// Single-token forward pass.
    ///
    /// token_id:     current input token (integer)
    /// pos:          position in sequence (0-indexed)
    /// decoded_so_far: decoded text up to this point (used by router for context)
    /// kv:           mutable KV cache (trunk + 4 expert slices)
    ///
    /// Returns (logits: Vec<i8>, expert_used: ExpertDomain).
    /// `logits` has length VOCAB_SIZE.
    pub fn forward(
        &self,
        token_id: u32,
        pos: usize,
        decoded_so_far: &str,
        kv: &mut KVCache,
    ) -> (Vec<i8>, ExpertDomain) {
        // 1. Shared trunk (all experts share this computation).
        let trunk_out = self.trunk.forward(token_id, pos, kv);

        // 2. Route to expert — deterministic, no learned gating.
        let domain = self.router.route_context(decoded_so_far);

        // 3. Expert forward (only one expert active per token).
        let expert_out = self.experts[domain.index()].forward(&trunk_out, pos, kv);

        // 4. LM head → vocab logits.
        let logits = self.lm_head.forward(&expert_out);

        debug_assert_eq!(logits.len(), VOCAB_SIZE);
        (logits, domain)
    }

    /// Construct from a parsed .s7l file.
    /// Wires weight tensors from the FIELD lane into model components.
    pub fn from_s7(_file: &crate::s7l::S7File) -> Self {
        // Full weight wiring requires the s7-pack-moe binary to have sealed
        // trained weights into the FIELD lane.  Until a real .s7l weight file
        // exists, constructing from scratch calls unimplemented!() to signal
        // the correct integration point.
        //
        // To wire:
        //   1. Open FIELD lane (id=2) from the parsed S7File.
        //   2. Iterate TensorRecords in deterministic name-sorted order.
        //   3. Match each tensor name to its component in the model.
        //   4. Replace zero-weight placeholders.
        unimplemented!(
            "Wire S7LlmMoe from .s7l FIELD lane (id=2). \
             Seal weights with: cargo run --bin s7-pack-moe -- \
             --weights-dir model/weights/ --vocab model/vocab.json \
             --out model/moe.s7l"
        )
    }
}
