/// Greedy (argmax) decoder for S7-LLM-MOE-300M.
///
/// Properties:
///   - No randomness: no temperature, top-p, top-k.
///   - Deterministic: same model + same prompt → same token sequence.
///   - V6 compliant: same inputs → same proof chain hash.
///   - Proof chain: SHA-256 chain over all step hashes (replayable).
///
/// Fold binding: ⟁COMPUTE_FOLD⟁ → BATCH lane.
/// CM-1 gate: U+0002 STX must be asserted by the caller (see main.rs).
use crate::model::S7LlmMoe;
use crate::inference::kv_cache::KVCache;
use crate::inference::proof::ProofChain;
use crate::tokenizer::bpe::Tokenizer;
use crate::model::router::argmax_i8;

/// Decoded output with proof chain.
pub struct DecodeResult {
    pub tokens:      Vec<u32>,
    pub proof_chain: ProofChain,
    /// Expert activation histogram over the full generation.
    pub expert_usage: [usize; 9],
}

/// Decode output token ids from a prompt, up to `max_new_tokens` steps.
///
/// Returns a DecodeResult with the full token sequence and proof chain.
pub fn decode(
    model:          &S7LlmMoe,
    tokenizer:      &Tokenizer,
    prompt_tokens:  Vec<u32>,
    max_new_tokens: usize,
) -> DecodeResult {
    let mut kv    = KVCache::new();
    let mut tokens = prompt_tokens.clone();
    let mut chain  = ProofChain::new();

    // Prefill: feed all prompt tokens through trunk + router + expert
    // to populate KV cache.
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let out = model.forward(tok, pos, &mut kv);
        chain.push(out.proof);
        // Discard logits during prefill (only the last one matters).
    }

    // Generation: autoregressive argmax decode.
    for _step in 0..max_new_tokens {
        let pos      = tokens.len() - 1;
        let last_tok = *tokens.last().unwrap();

        let out      = model.forward(last_tok, pos, &mut kv);
        let next_tok = argmax_i8(&out.logits) as u32;

        chain.push(out.proof);

        // EOS: token 1 is conventional EOS for 32k BPE.
        if next_tok == 1 {
            break;
        }

        tokens.push(next_tok);
        let _ = tokenizer; // tokenizer available for streaming decode if needed
    }

    let expert_usage = chain.expert_usage();
    DecodeResult { tokens, proof_chain: chain, expert_usage }
}
