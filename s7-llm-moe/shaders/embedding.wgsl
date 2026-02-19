// embedding.wgsl
// Token embedding lookup for WebGPU path.
//
// Embedding table stored as INT8 (dequantised on-the-fly).
// Shape: [vocab_size, hidden_dim] packed as i32 (4 bytes per word).
//
// Bind group:
//   binding(0) : array<i32>   emb_table  [vocab_size * hidden_dim / 4]
//   binding(1) : array<f32>   emb_scale  [vocab_size] (one scale per token)
//   binding(2) : array<f32>   output     [hidden_dim]
//   binding(3) : EmbUniforms  { token_id, hidden_dim }

struct EmbUniforms {
    token_id   : u32,
    hidden_dim : u32,
}

@group(0) @binding(0) var<storage, read>       emb_table : array<i32>;
@group(0) @binding(1) var<storage, read>       emb_scale : array<f32>;
@group(0) @binding(2) var<storage, read_write> output    : array<f32>;
@group(0) @binding(3) var<uniform>             uniforms  : EmbUniforms;

fn unpack_i8(packed: i32, lane: u32) -> f32 {
    let shift  = lane * 8u;
    let masked = (packed >> shift) & 0xFF;
    let signed = select(masked, masked - 256, masked >= 128);
    return f32(signed);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim_idx = gid.x;
    if dim_idx >= uniforms.hidden_dim { return; }

    let hd        = uniforms.hidden_dim;
    let words_row = (hd + 3u) / 4u;
    let row_start = uniforms.token_id * words_row;
    let word_idx  = dim_idx / 4u;
    let lane      = dim_idx % 4u;

    let packed = emb_table[row_start + word_idx];
    let scale  = emb_scale[uniforms.token_id];

    output[dim_idx] = unpack_i8(packed, lane) * scale;
}
