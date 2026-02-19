/// INT8-quantized tensor.
///
/// All weight tensors in S7-LLM-MOE are stored in INT8 and operate
/// in INT8/INT32 arithmetic throughout inference.  No FP32 is used
/// for weight matrix multiplications; scale factors are applied only
/// when converting final logits to token probabilities (which are
/// never computed — we use argmax directly).
///
/// Memory layout:
///   data is stored row-major, padded to a 32-byte (AVX2) boundary.
///   padding bytes are 0x00 and do not affect computation.
///
/// Dequantisation (reference only, not used in deterministic path):
///   val_f32 = data[i] as f32 * scale
#[derive(Clone)]
pub struct Int8Tensor {
    pub name:  String,
    pub dims:  Vec<usize>,
    pub scale: f32,
    /// Padded to 32-byte boundary (AVX2 alignment).
    pub data:  Vec<i8>,
}

impl Int8Tensor {
    /// Logical element count (without AVX2 padding).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Byte count including AVX2 padding.
    pub fn padded_len(&self) -> usize {
        avx2_pad(self.numel())
    }

    /// Return a zero tensor with given dims (used for weight-init placeholders).
    pub fn zeros(name: impl Into<String>, dims: Vec<usize>, scale: f32) -> Self {
        let n = dims.iter().product::<usize>();
        let padded = avx2_pad(n);
        Self {
            name:  name.into(),
            dims,
            scale,
            data:  vec![0i8; padded],
        }
    }

    /// Matrix-vector multiply: out[j] = Σ_i weight[i,j] × input[i]
    ///
    /// weight.dims = [in_dim, out_dim]
    /// Returns vec of length out_dim, each element in i8 (saturating).
    pub fn matvec(&self, input: &[i8]) -> Vec<i8> {
        debug_assert_eq!(self.dims.len(), 2, "matvec requires 2-D tensor");
        let rows = self.dims[0]; // in_dim
        let cols = self.dims[1]; // out_dim
        debug_assert_eq!(input.len(), rows);

        #[cfg(has_avx2)]
        {
            avx2::matvec_avx2(&self.data, input, rows, cols)
        }
        #[cfg(not(has_avx2))]
        {
            crate::tensor::avx2::matvec_scalar(&self.data, input, rows, cols)
        }
    }
}

/// Round up `n` to the next multiple of 32 (AVX2 register width in i8 lanes).
pub fn avx2_pad(n: usize) -> usize {
    (n + 31) & !31
}
