// attention.wgsl
// Causal single-token attention for S7-LLM-MOE WebGPU path.
//
// Inputs are f32 (dequantised from INT8 by the matmul shader).
// This shader computes:
//   scores[t] = Q · K[t] / sqrt(head_dim)   for t in [0, seq_len)
//   weights[t] = softmax(scores)[t]
//   context    = sum_t(weights[t] * V[t])
//
// One dispatch per head.
//
// Bind group:
//   binding(0) : array<f32>  q_vec      [head_dim]
//   binding(1) : array<f32>  k_cache    [seq_len * head_dim]
//   binding(2) : array<f32>  v_cache    [seq_len * head_dim]
//   binding(3) : array<f32>  context    [head_dim]   (output)
//   binding(4) : Uniforms    { head_dim, seq_len }

struct AttnUniforms {
    head_dim : u32,
    seq_len  : u32,
}

@group(0) @binding(0) var<storage, read>       q_vec   : array<f32>;
@group(0) @binding(1) var<storage, read>       k_cache : array<f32>;
@group(0) @binding(2) var<storage, read>       v_cache : array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx_out : array<f32>;
@group(0) @binding(4) var<uniform>             uniforms: AttnUniforms;

var<workgroup> scores: array<f32, 2048>;  // max seq len

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let hd  = uniforms.head_dim;
    let sl  = uniforms.seq_len;
    let tid = lid.x;

    // Phase 1: each thread computes scores for a stride of seq positions.
    let scale = 1.0 / sqrt(f32(hd));
    var t = tid;
    loop {
        if t >= sl { break; }
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < hd; d++) {
            dot += q_vec[d] * k_cache[t * hd + d];
        }
        scores[t] = dot * scale;
        t += 64u;
    }
    workgroupBarrier();

    // Phase 2: thread 0 computes softmax and weighted V sum.
    if tid == 0u {
        // Find max for numerical stability.
        var max_s: f32 = scores[0];
        for (var i: u32 = 1u; i < sl; i++) {
            max_s = max(max_s, scores[i]);
        }
        // Compute exp weights and sum.
        var w_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < sl; i++) {
            scores[i] = exp(scores[i] - max_s);
            w_sum += scores[i];
        }
        // Weighted V sum → context output.
        for (var d: u32 = 0u; d < hd; d++) {
            var ctx: f32 = 0.0;
            for (var i: u32 = 0u; i < sl; i++) {
                ctx += (scores[i] / w_sum) * v_cache[i * hd + d];
            }
            ctx_out[d] = ctx;
        }
    }
}
