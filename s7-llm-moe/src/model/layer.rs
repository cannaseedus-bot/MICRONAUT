/// A single transformer layer: Attention + FFN with residual connections.
///
/// Operates in INT8 throughout.  Residual connections add i8 + i8 with
/// saturation clamp (no overflow).
use super::attention::MultiHeadAttention;
use super::ffn::FFN;
use super::rope::RopeTable;
use crate::inference::kv_cache::KVLayer;

pub struct TransformerLayer {
    pub attn: MultiHeadAttention,
    pub ffn:  FFN,
}

impl TransformerLayer {
    /// Forward pass for one transformer layer at sequence position `pos`.
    ///
    /// hidden_in: i8 vec of length hidden_dim
    /// Returns:   i8 vec of length hidden_dim
    pub fn forward(
        &self,
        hidden_in: &[i8],
        pos: usize,
        kv: &mut KVLayer,
        rope: &RopeTable,
    ) -> Vec<i8> {
        // Pre-norm is approximated by a layer-norm free pass (INT8 scale keeps norms stable).
        // Attention sublayer with residual.
        let attn_out = self.attn.forward(hidden_in, pos, kv, rope);
        let h1 = residual_add(hidden_in, &attn_out);

        // FFN sublayer with residual.
        let ffn_out = self.ffn.forward(&h1);
        residual_add(&h1, &ffn_out)
    }
}

/// Element-wise saturating i8 residual addition.
#[inline]
fn residual_add(a: &[i8], b: &[i8]) -> Vec<i8> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            ((ai as i16 + bi as i16).clamp(-128, 127)) as i8
        })
        .collect()
}
