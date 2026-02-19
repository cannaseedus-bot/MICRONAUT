/// BPE Tokenizer for the 24,576-entry vocabulary.
///
/// Vocabulary is stored in the DICT lane (SCXQ2 id=1) of the .s7l file
/// as a flat list of (token_id → UTF-8 string) mappings.
///
/// For the runtime binary, we also support loading from a JSON file
/// (key = token_id as string, value = token string) matching the pattern
/// used by s7-llm-mini.
use serde_json::Value;
use std::collections::HashMap;
use std::fs;

pub struct Tokenizer {
    /// id → decoded string
    pub id_to_str: HashMap<u32, String>,
    /// string → id (for encoding)
    pub str_to_id: HashMap<String, u32>,
}

impl Tokenizer {
    /// Load from a JSON vocab file.
    /// Format: { "token_string": token_id_int, ... }
    pub fn from_file(path: &str) -> Self {
        let raw = fs::read_to_string(path).expect("cannot read vocab file");
        let parsed: Value = serde_json::from_str(&raw).expect("invalid JSON vocab");
        let obj = parsed.as_object().expect("vocab must be a JSON object");

        let mut id_to_str = HashMap::with_capacity(obj.len());
        let mut str_to_id = HashMap::with_capacity(obj.len());

        for (token_str, id_val) in obj {
            let id = id_val.as_u64().expect("token id must be integer") as u32;
            id_to_str.insert(id, token_str.clone());
            str_to_id.insert(token_str.clone(), id);
        }

        Self { id_to_str, str_to_id }
    }

    /// Load from the DICT lane payload of a .s7l file.
    /// DICT lane format: repeated records of (u16 str_len, [u8] str, u32 id).
    pub fn from_dict_lane(payload: &[u8]) -> Self {
        let mut id_to_str = HashMap::new();
        let mut str_to_id = HashMap::new();
        let mut i = 0;

        while i + 6 <= payload.len() {
            let str_len = u16::from_be_bytes(payload[i..i + 2].try_into().unwrap()) as usize;
            i += 2;
            if i + str_len + 4 > payload.len() { break; }
            let token_str = std::str::from_utf8(&payload[i..i + str_len])
                .unwrap_or("")
                .to_string();
            i += str_len;
            let id = u32::from_be_bytes(payload[i..i + 4].try_into().unwrap());
            i += 4;

            id_to_str.insert(id, token_str.clone());
            str_to_id.insert(token_str, id);
        }

        Self { id_to_str, str_to_id }
    }

    /// Encode a string to token ids using greedy longest-match.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut pos = 0;
        let chars: Vec<char> = text.chars().collect();

        while pos < chars.len() {
            // Greedy longest-match from position `pos`.
            let mut best_len = 1;
            let mut best_id  = *self.str_to_id.get("▁").unwrap_or(&0); // UNK

            for end in (pos + 1..=chars.len()).rev() {
                let candidate: String = chars[pos..end].iter().collect();
                if let Some(&id) = self.str_to_id.get(&candidate) {
                    best_len = end - pos;
                    best_id  = id;
                    break;
                }
            }

            tokens.push(best_id);
            pos += best_len;
        }

        tokens
    }

    /// Decode a sequence of token ids to a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&id| self.decode_single(id)).collect()
    }

    /// Decode a single token id.
    pub fn decode_single(&self, id: u32) -> String {
        self.id_to_str
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("<unk:{}>", id))
    }
}
