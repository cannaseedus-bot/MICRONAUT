// matmul_int8.wgsl
// INT8 matrix-vector multiply for S7-LLM-MOE WebGPU path.
//
// WebGPU does not have native INT8 MAD instructions; we widen to i32.
// Dequantisation: val_f32 = f32(data_i8) * scale  (on the fly, per row)
//
// Workgroup tile: 16 output elements per invocation group.
// Each invocation computes one output element (one row dot-product).
//
// Bind group layout:
//   binding(0) : array<i32>  weight_i8  (packed as i32, 4 bytes = 4 i8 lanes)
//   binding(1) : array<i32>  input_i8   (packed as i32)
//   binding(2) : array<f32>  scales     (one per output row)
//   binding(3) : array<f32>  output_f32 (result, one f32 per output)
//   binding(4) : Uniforms    { in_dim: u32, out_dim: u32 }

struct Uniforms {
    in_dim  : u32,
    out_dim : u32,
}

@group(0) @binding(0) var<storage, read>       weight_i8  : array<i32>;
@group(0) @binding(1) var<storage, read>       input_i8   : array<i32>;
@group(0) @binding(2) var<storage, read>       scales     : array<f32>;
@group(0) @binding(3) var<storage, read_write> output_f32 : array<f32>;
@group(0) @binding(4) var<uniform>             uniforms   : Uniforms;

// Extract the k-th i8 lane from a packed i32 (little-endian byte order).
fn unpack_i8(packed: i32, lane: u32) -> i32 {
    let shift  = lane * 8u;
    let masked = (packed >> shift) & 0xFF;
    // Sign-extend from 8 bits.
    return select(masked, masked - 256, masked >= 128);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if out_idx >= uniforms.out_dim { return; }

    let in_words  = (uniforms.in_dim + 3u) / 4u;  // ceil(in_dim / 4) packed i32 words
    let row_start = out_idx * in_words;

    var acc: i32 = 0;

    // Dot-product: weight[out_idx, :] · input[:]
    for (var w: u32 = 0u; w < in_words; w++) {
        let w_pack = weight_i8[row_start + w];
        let x_pack = input_i8[w];

        // Unpack 4 INT8 pairs and accumulate.
        let pairs = min(4u, uniforms.in_dim - w * 4u);
        for (var lane: u32 = 0u; lane < pairs; lane++) {
            acc += unpack_i8(w_pack, lane) * unpack_i8(x_pack, lane);
        }
    }

    // Dequantise: f32 output = i32_acc * row_scale
    output_f32[out_idx] = f32(acc) * scales[out_idx];
}
