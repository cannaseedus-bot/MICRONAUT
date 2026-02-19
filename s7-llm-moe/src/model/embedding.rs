use crate::tensor::Int8Tensor;
use crate::model::config::VOCAB_SIZE;

/// Token embedding table.
/// Shape: [VOCAB_SIZE, hidden_dim]
/// Weight-tied with the LM head (shared pointer in full implementation).
pub struct Embedding {
    pub weight: Int8Tensor, // [vocab_size, hidden_dim]
}

impl Embedding {
    /// Look up the embedding vector for a single token id.
    /// Returns a slice of length hidden_dim in i8.
    pub fn lookup(&self, token_id: u32) -> Vec<i8> {
        let id = token_id as usize;
        debug_assert!(id < VOCAB_SIZE, "token id {} out of vocab range", id);
        let hidden = self.weight.dims[1];
        let start  = id * hidden;
        self.weight.data[start..start + hidden].to_vec()
    }
}
