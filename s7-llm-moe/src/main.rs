mod s7l;
mod tensor;
mod model;
mod tokenizer;
mod inference;
mod webgpu;

use std::env;
use std::fs;

use s7l::S7File;
use tokenizer::bpe::Tokenizer;
use model::S7LlmMoe;
use inference::greedy::decode;
use model::config::DEFAULT_MAX_TOKENS;

/// CM-1 control phase gating.
/// U+0002 STX = @control.body.begin — must be asserted before inference.
/// U+0004 EOT = @control.transmission.end — asserted after generation.
const CM1_STX: char = '\u{0002}';
const CM1_EOT: char = '\u{0004}';

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse CLI flags.
    let model_path = flag_val(&args, "--model").unwrap_or("model/moe.s7l");
    let vocab_path = flag_val(&args, "--vocab").unwrap_or("model/vocab.json");
    let prompt     = flag_val(&args, "--prompt").unwrap_or("hello");
    let max_tokens = flag_val(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    // CM-1 gate: assert STX before any inference.
    print!("{}", CM1_STX);

    // Load .s7l sealed artifact.
    let model_bytes = fs::read(model_path)
        .unwrap_or_else(|_| panic!("cannot read model file: {}", model_path));
    let s7 = S7File::parse(&model_bytes)
        .unwrap_or_else(|e| panic!("s7l parse error: {}", e));

    // Load tokenizer.
    let tokenizer = Tokenizer::from_file(vocab_path);

    // Construct model from .s7l (wires FIELD lane weights).
    let model = S7LlmMoe::from_s7(&s7);

    // Encode prompt.
    let prompt_tokens = tokenizer.encode(prompt);

    eprintln!(
        "[S7-LLM-MOE] model={} vocab={} prompt_tokens={} max_new={}",
        model_path, vocab_path, prompt_tokens.len(), max_tokens
    );

    // Greedy decode — deterministic, V6 compliant.
    let output_tokens = decode(&model, &tokenizer, prompt_tokens, max_tokens);
    let output_text   = tokenizer.decode(&output_tokens);

    // CM-1 gate: assert EOT after generation completes.
    print!("{}", CM1_EOT);

    println!("{}", output_text);
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(&args[i + 1]);
        }
    }
    None
}
