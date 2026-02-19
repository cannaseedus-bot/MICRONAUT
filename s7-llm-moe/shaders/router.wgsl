// router.wgsl
// Deterministic token routing for WebGPU path.
//
// This shader encodes the lexical routing table as a lookup over a
// precomputed trigger-domain bitmap.  The CPU deterministic router result
// is passed in as a uniform — this shader is provided for completeness
// so the full pipeline can run on GPU.
//
// In practice the DeterministicRouter runs CPU-side (it processes text,
// not tensors) and the result is uploaded as a single u32 uniform.
//
// Bind group:
//   binding(0) : RouteUniforms { expert_id: u32 }
//   binding(1) : array<f32>    input      [hidden] (trunk output)
//   binding(2) : array<f32>    output     [hidden] (passed to active expert)
//
// The shader is a simple identity copy — routing selection happens CPU-side.
// The GPU path selects the correct expert weight buffers before dispatching.

struct RouteUniforms {
    expert_id  : u32,     // 0=Code, 1=Math, 2=Reason, 3=General
    hidden_dim : u32,
}

@group(0) @binding(0) var<uniform>             route  : RouteUniforms;
@group(0) @binding(1) var<storage, read>       input  : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= route.hidden_dim { return; }
    // Pass trunk output through to the selected expert input buffer.
    output[idx] = input[idx];
}
