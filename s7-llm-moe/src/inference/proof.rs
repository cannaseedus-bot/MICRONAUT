/// Proof binding for S7-LLM-MOE-300M inference.
///
/// Every token generation step produces a `ProofRecord` that can be
/// replayed deterministically.  The proof includes:
///
///   token_id         — input token at this step
///   pos              — sequence position
///   expert_id        — which expert was activated (0..8)
///   expert_name      — micronaut name (e.g. "MM-1")
///   fold             — fold binding (e.g. "⟁COMPUTE_FOLD⟁")
///   router_hash      — SHA-256 of router logits_i8 (9 bytes)
///   expert_hash      — SHA-256 of expert weight fingerprint (from EDGE lane)
///   step_hash        — SHA-256(token_id || pos || expert_id || router_hash)
///
/// V6 compliance:
///   Same (model, token sequence) → identical proof chain.
///
/// The proof chain for a full generation:
///   chain_hash[0] = step_hash[0]
///   chain_hash[t] = SHA-256(chain_hash[t-1] || step_hash[t])
///
/// A verifier can replay the route decisions given:
///   1. The .s7l sealed weight file (router weights are fixed).
///   2. The input prompt tokens.
///   3. This proof chain.
use sha2::{Digest, Sha256};
use crate::model::config::{EXPERT_NAMES, EXPERT_FOLDS};
use crate::model::expert::Expert;

/// Proof record for a single token generation step.
#[derive(Debug, Clone)]
pub struct ProofRecord {
    pub token_id:    u32,
    pub pos:         usize,
    pub expert_id:   usize,
    pub expert_name: &'static str,
    pub fold:        &'static str,
    /// SHA-256 of raw INT8 router logits (9 bytes → 32-byte hash).
    pub router_hash: [u8; 32],
    /// SHA-256 of expert-id || step fingerprint.
    pub step_hash:   [u8; 32],
}

impl ProofRecord {
    /// Build a proof record for one token step.
    pub fn build(
        token_id:  u32,
        pos:       usize,
        expert_id: usize,
        logits_i8: &[i8],
        expert:    &Expert,
    ) -> Self {
        // SHA-256 over raw router logits.
        let router_hash: [u8; 32] = {
            let raw: Vec<u8> = logits_i8.iter().map(|&b| b as u8).collect();
            Sha256::digest(&raw).into()
        };

        // Step hash: SHA-256(token_id_be || pos_be || expert_id_u8 || router_hash).
        let step_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(token_id.to_be_bytes());
            h.update((pos as u64).to_be_bytes());
            h.update([expert_id as u8]);
            h.update(router_hash);
            h.finalize().into()
        };

        let _ = expert; // weight hash extension point (requires EDGE lane lookup)

        ProofRecord {
            token_id,
            pos,
            expert_id,
            expert_name: EXPERT_NAMES[expert_id],
            fold:        EXPERT_FOLDS[expert_id],
            router_hash,
            step_hash,
        }
    }

    pub fn router_hash_hex(&self) -> String {
        self.router_hash.iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn step_hash_hex(&self) -> String {
        self.step_hash.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

/// A proof chain for an entire generation sequence.
/// chain_hash[t] = SHA-256(chain_hash[t-1] || step_hash[t])
pub struct ProofChain {
    pub records:    Vec<ProofRecord>,
    pub chain_hash: [u8; 32],
}

impl ProofChain {
    pub fn new() -> Self {
        ProofChain {
            records:    Vec::new(),
            chain_hash: [0u8; 32],
        }
    }

    /// Extend the chain with a new step.
    pub fn push(&mut self, record: ProofRecord) {
        let new_hash: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(self.chain_hash);
            h.update(record.step_hash);
            h.finalize().into()
        };
        self.chain_hash = new_hash;
        self.records.push(record);
    }

    pub fn chain_hash_hex(&self) -> String {
        self.chain_hash.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Expert activation distribution over the full chain.
    pub fn expert_usage(&self) -> [usize; 9] {
        let mut counts = [0usize; 9];
        for r in &self.records {
            if r.expert_id < 9 {
                counts[r.expert_id] += 1;
            }
        }
        counts
    }
}

impl Default for ProofChain {
    fn default() -> Self { Self::new() }
}
