/// Greedy (argmax) decoder for S7-LLM-MOE-140M.
///
/// Properties guaranteed by this implementation:
///   - No randomness (no temperature, top-p, top-k).
///   - Identical output for identical input on any platform.
///   - V6 compliant: same tokens + same model → same output hash.
///   - CM-1 gate: U+0002 STX must be asserted by the caller before invoking.
///
/// Fold binding: ⟁COMPUTE_FOLD⟁ → BATCH lane.
use crate::model::S7LlmMoe;
use crate::inference::kv_cache::KVCache;
use crate::tokenizer::bpe::Tokenizer;

/// Decode output token ids from a prompt, up to `max_new_tokens` steps.
///
/// Returns the full token sequence (prompt tokens + generated tokens).
pub fn decode(
    model:          &S7LlmMoe,
    tokenizer:      &Tokenizer,
    prompt_tokens:  Vec<u32>,
    max_new_tokens: usize,
) -> Vec<u32> {
    let mut kv    = KVCache::new();
    let mut tokens = prompt_tokens.clone();
    let mut decoded_text = String::new();

    // Prefill: feed all prompt tokens through the model to populate KV cache.
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let tok_str = tokenizer.decode_single(tok);
        decoded_text.push_str(&tok_str);
        let (_logits, _domain) = model.forward(tok, pos, &decoded_text, &mut kv);
        // During prefill we discard logits (only need the final step for next token).
    }

    // Generation: autoregressively produce new tokens.
    for step in 0..max_new_tokens {
        let pos       = tokens.len() - 1;
        let last_tok  = *tokens.last().unwrap();

        let (logits, _domain) = model.forward(last_tok, pos, &decoded_text, &mut kv);

        // Greedy argmax: no randomness, fully deterministic.
        let next_tok = argmax_u32(&logits) as u32;

        // EOS check (token 1 is conventional EOS for 24k BPE).
        if next_tok == 1 {
            break;
        }

        tokens.push(next_tok);
        let tok_str = tokenizer.decode_single(next_tok);
        decoded_text.push_str(&tok_str);

        let _ = step; // suppress lint
    }

    tokens
}

/// Return the index of the maximum i8 value (greedy argmax).
/// Ties broken by lowest index (deterministic).
fn argmax_u32(logits: &[i8]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap_or(0)
}
