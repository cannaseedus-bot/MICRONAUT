// ffn.wgsl
// Feed-Forward Network with GELU activation for WebGPU path.
//
// FFN(x) = W2 · GELU(W1 · x)
// GELU(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// W1 and W2 weights are pre-dequantised to f32 by matmul_int8.wgsl.
// This shader applies GELU to the intermediate activations and the
// second linear projection.
//
// Bind group:
//   binding(0) : array<f32>  intermediate   [ffn_dim]  (W1·x output)
//   binding(1) : array<f32>  w2_out         [hidden]   (W2·GELU(x) output)
//   binding(2) : array<f32>  w2_weights     [ffn_dim * hidden]
//   binding(3) : FfnUniforms { ffn_dim, hidden_dim }

struct FfnUniforms {
    ffn_dim    : u32,
    hidden_dim : u32,
}

@group(0) @binding(0) var<storage, read>       intermediate : array<f32>;
@group(0) @binding(1) var<storage, read_write> w2_out       : array<f32>;
@group(0) @binding(2) var<storage, read>       w2_weights   : array<f32>;
@group(0) @binding(3) var<uniform>             uniforms     : FfnUniforms;

// GELU activation (Hendrycks & Gimpel, 2016).
fn gelu(x: f32) -> f32 {
    let k = 0.7978845608028654;    // sqrt(2/π)
    let c = 0.044715;
    return x * 0.5 * (1.0 + tanh(k * (x + c * x * x * x)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if out_idx >= uniforms.hidden_dim { return; }

    var acc: f32 = 0.0;
    for (var f: u32 = 0u; f < uniforms.ffn_dim; f++) {
        // Apply GELU to intermediate activations from W1 pass.
        let act = gelu(intermediate[f]);
        // W2 projection.
        acc += act * w2_weights[f * uniforms.hidden_dim + out_idx];
    }

    w2_out[out_idx] = acc;
}
