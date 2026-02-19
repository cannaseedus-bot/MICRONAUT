/// Learned Router — S7-LLM-MOE-300M
///
/// Replaces the lexical DeterministicRouter from the 140M design.
/// The router is a 2-layer MLP trained end-to-end with the model:
///
///   h_shared ∈ ℝ^1024   (trunk hidden state at last layer)
///       ↓
///   fc1: Linear(1024 → 2048), INT8 at inference
///       ↓
///   GELU
///       ↓
///   fc2: Linear(2048 → 9), INT8 at inference
///       ↓
///   router_logits ∈ ℝ^9   (one logit per expert/micronaut)
///       ↓
///   expert_id = argmax(router_logits)   ← DETERMINISTIC at inference
///
/// Training objective (Python training loop):
///   L_total = L_lm + α * L_balance + β * L_entropy
///
/// The argmax selection at inference preserves V6 determinism:
///   same h_shared + same router weights → same expert_id, always.
///
/// Proof binding: SHA-256 of router_logits_i8 is included in the
/// inference proof record (see inference/proof.rs).
use super::linear::Linear;
use super::ffn::gelu_i8;
use crate::model::config::{NUM_EXPERTS, EXPERT_NAMES};

/// The learned router MLP.
pub struct LearnedRouter {
    pub fc1: Linear,   // [TRUNK_HIDDEN, ROUTER_HIDDEN]
    pub fc2: Linear,   // [ROUTER_HIDDEN, NUM_EXPERTS]
}

/// Result of one router forward pass.
#[derive(Debug, Clone)]
pub struct RouterOutput {
    /// Chosen expert index (0..NUM_EXPERTS). Deterministic: argmax of logits.
    pub expert_id:   usize,
    /// Raw INT8 router logits before argmax (included in proof hash).
    pub logits_i8:   Vec<i8>,
    /// Micronaut name for the chosen expert.
    pub expert_name: &'static str,
    /// Fold binding for the chosen expert.
    pub fold:        &'static str,
}

impl LearnedRouter {
    /// Forward pass.
    ///
    /// h_shared: trunk hidden state, i8 vector of length TRUNK_HIDDEN.
    /// Returns RouterOutput with deterministic expert_id.
    pub fn forward(&self, h_shared: &[i8]) -> RouterOutput {
        // Layer 1: fc1 + GELU
        let h = self.fc1.forward(h_shared);
        let h = gelu_i8(&h);

        // Layer 2: fc2 → expert logits
        let logits_i8 = self.fc2.forward(&h);
        debug_assert_eq!(logits_i8.len(), NUM_EXPERTS);

        // Deterministic argmax — no sampling, no temperature.
        let expert_id = argmax_i8(&logits_i8);

        RouterOutput {
            expert_id,
            logits_i8,
            expert_name: EXPERT_NAMES[expert_id],
            fold:        crate::model::config::EXPERT_FOLDS[expert_id],
        }
    }
}

/// Argmax over INT8 logits. Ties broken by lowest index (deterministic).
#[inline]
pub fn argmax_i8(logits: &[i8]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Softmax over i8 logits → f32 probabilities.
/// Used only for building the proof record (not for routing selection).
pub fn softmax_f32(logits_i8: &[i8]) -> Vec<f32> {
    let max_v = logits_i8.iter().copied().max().unwrap_or(0) as f32;
    let exps: Vec<f32> = logits_i8
        .iter()
        .map(|&x| ((x as f32) - max_v).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Int8Tensor;

    fn zero_router() -> LearnedRouter {
        LearnedRouter {
            fc1: super::super::linear::Linear {
                weight: Int8Tensor::zeros(
                    "router.fc1.weight",
                    vec![crate::model::config::TRUNK_HIDDEN,
                         crate::model::config::ROUTER_HIDDEN],
                    1.0 / 127.0,
                ),
            },
            fc2: super::super::linear::Linear {
                weight: Int8Tensor::zeros(
                    "router.fc2.weight",
                    vec![crate::model::config::ROUTER_HIDDEN, NUM_EXPERTS],
                    1.0 / 127.0,
                ),
            },
        }
    }

    #[test]
    fn router_output_length() {
        let r = zero_router();
        let h = vec![0i8; crate::model::config::TRUNK_HIDDEN];
        let out = r.forward(&h);
        assert_eq!(out.logits_i8.len(), NUM_EXPERTS);
        assert!(out.expert_id < NUM_EXPERTS);
    }

    #[test]
    fn router_is_deterministic() {
        let r = zero_router();
        let h = vec![1i8; crate::model::config::TRUNK_HIDDEN];
        let a = r.forward(&h);
        let b = r.forward(&h);
        assert_eq!(a.expert_id, b.expert_id);
        assert_eq!(a.logits_i8, b.logits_i8);
    }
}
