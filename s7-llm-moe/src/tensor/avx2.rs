/// AVX2 and scalar INT8 matrix-vector multiply kernels.
///
/// AVX2 layout:
///   Weight matrix stored row-major, rows padded to 32-byte (256-bit) alignment.
///   Each AVX2 iteration loads 32 i8 weight values and 32 i8 input values,
///   computing a partial dot product via `_mm256_maddubs_epi16` + horizontal add.
///
/// Scalar fallback:
///   Identical arithmetic but using plain Rust loops.
///   Used when AVX2 is not available (WebAssembly, ARM, etc.).

/// Scalar INT8 matvec (reference implementation, always available).
///
/// weight_data: row-major, rows padded to 32 bytes.
/// Returns out_dim i8 values (i32 accumulator, saturating shift-right by 7).
pub fn matvec_scalar(
    weight_data: &[i8],
    input: &[i8],
    in_dim: usize,
    out_dim: usize,
) -> Vec<i8> {
    // Stride in the stored weight matrix includes AVX2 padding.
    let row_stride = avx2_pad_usize(in_dim);
    let mut out = vec![0i32; out_dim];

    for col in 0..out_dim {
        let mut acc = 0i32;
        for row in 0..in_dim {
            // Weight layout: weight[col, row] stored at data[col * row_stride + row]
            // (transposed for column-major access on output side)
            acc += (weight_data[col * row_stride + row] as i32)
                 * (input[row] as i32);
        }
        // Arithmetic right-shift by 7 to keep in i8 range.
        out[col] = acc;
    }

    // Convert i32 → i8 with saturation.
    out.iter().map(|&v| (v >> 7).clamp(-128, 127) as i8).collect()
}

/// AVX2 INT8 matvec — compiled only on x86_64 with +avx2.
#[cfg(has_avx2)]
pub mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    pub fn matvec_avx2(
        weight_data: &[i8],
        input: &[i8],
        in_dim: usize,
        out_dim: usize,
    ) -> Vec<i8> {
        let row_stride = super::avx2_pad_usize(in_dim);
        let mut out_i32 = vec![0i32; out_dim];

        unsafe {
            for col in 0..out_dim {
                let w_ptr = weight_data[col * row_stride..].as_ptr();
                let x_ptr = input.as_ptr();
                let mut acc = _mm256_setzero_si256();

                let blocks = in_dim / 32;
                for b in 0..blocks {
                    let w_vec = _mm256_loadu_si256(
                        w_ptr.add(b * 32) as *const __m256i
                    );
                    let x_vec = _mm256_loadu_si256(
                        x_ptr.add(b * 32) as *const __m256i
                    );
                    // _mm256_maddubs_epi16: u8 × i8 → i16 pairs, saturating
                    // We reinterpret i8 weights as i8 and use the signed-unsigned form.
                    let prod = _mm256_maddubs_epi16(
                        // Convert i8 → u8 by bias (shift +128), undo bias in accumulator
                        _mm256_add_epi8(w_vec, _mm256_set1_epi8(-128i8)),
                        x_vec,
                    );
                    // Widen i16 pairs to i32 and accumulate.
                    let prod_i32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
                    acc = _mm256_add_epi32(acc, prod_i32);
                }

                // Horizontal sum of 8 i32 lanes.
                let sum_lo = _mm256_extracti128_si256(acc, 0);
                let sum_hi = _mm256_extracti128_si256(acc, 1);
                let sum128 = _mm_add_epi32(sum_lo, sum_hi);
                let shuf   = _mm_shuffle_epi32(sum128, 0b_10_11_00_01);
                let sum64  = _mm_add_epi32(sum128, shuf);
                let shuf2  = _mm_shuffle_epi32(sum64, 0b_01_00_11_10);
                let total  = _mm_add_epi32(sum64, shuf2);
                let mut scalar_acc = _mm_cvtsi128_si32(total);

                // Scalar tail (remaining elements after last full 32-wide block).
                let w_tail = &weight_data[col * row_stride + blocks * 32..];
                let x_tail = &input[blocks * 32..in_dim];
                for (w, x) in w_tail.iter().zip(x_tail.iter()) {
                    scalar_acc += (*w as i32) * (*x as i32);
                }

                out_i32[col] = scalar_acc;
            }
        }

        out_i32.iter().map(|&v| (v >> 7).clamp(-128, 127) as i8).collect()
    }
}

#[cfg(has_avx2)]
pub use avx2::matvec_avx2;

pub fn avx2_pad_usize(n: usize) -> usize {
    (n + 31) & !31
}
