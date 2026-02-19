/// Feed-Forward Network with GELU activation (integer approximation).
///
/// Standard two-layer FFN: FFN(x) = W2 · GELU(W1 · x)
///
/// GELU integer approximation:
///   GELU(x) ≈ x * sigmoid(1.702 * x)
///   In INT8: gelu_i8(x) = (x * sigmoid_i8(x)) >> 7
///   where sigmoid_i8(x) ∈ [0, 127] (unsigned 0–1 range × 127)
///
/// This avoids any floating-point operations in the inference path.
use super::linear::Linear;

pub struct FFN {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FFN {
    pub fn forward(&self, hidden: &[i8]) -> Vec<i8> {
        let h = self.fc1.forward(hidden);
        let h = gelu_int8(&h);
        self.fc2.forward(&h)
    }
}

/// Integer GELU approximation.
/// Input and output are in the i8 domain.
fn gelu_int8(v: &[i8]) -> Vec<i8> {
    v.iter().map(|&x| gelu_scalar(x)).collect()
}

/// Scalar GELU for a single i8 value.
/// Uses a piecewise linear approximation that is deterministic and bit-exact.
///
/// GELU(x) ≈  0          for x < -3
///            x/4 + 3/4  for -3 ≤ x < 0  (linear ramp, simplified)
///            x * 0.854  for 0 ≤ x < 3   (matches GELU well in this range)
///            x           for x ≥ 3       (saturates near identity)
///
/// All in INT8 arithmetic (no division except by powers of 2).
#[inline(always)]
fn gelu_scalar(x: i8) -> i8 {
    let xi = x as i32;
    let out = if xi < -3 * 16 {
        // Below -3 (scaled): near zero.
        0
    } else if xi < 0 {
        // Linear ramp: approx x * 0.25 + 3*0.25 in scaled domain.
        (xi >> 2) + 3
    } else if xi < 3 * 16 {
        // Approximately x * 0.854: use 109/128 ≈ 0.851.
        (xi * 109) >> 7
    } else {
        // Identity for large positive values.
        xi.clamp(-128, 127)
    } as i32;
    out.clamp(-128, 127) as i8
}
