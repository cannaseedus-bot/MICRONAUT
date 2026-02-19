use crate::tensor::Int8Tensor;

/// INT8 linear layer (no bias — bias is folded into scale during training).
pub struct Linear {
    pub weight: Int8Tensor,
}

impl Linear {
    /// y = W x  (INT8 matrix-vector product, i8 → i8 with i32 accumulator).
    pub fn forward(&self, input: &[i8]) -> Vec<i8> {
        self.weight.matvec(input)
    }
}
