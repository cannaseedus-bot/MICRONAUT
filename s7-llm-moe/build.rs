// build.rs — emit AVX2 cfg flag when the target supports it.
// The model tensor layout is AVX2-aligned (32-byte boundaries) regardless;
// this flag gates whether the runtime matmul uses AVX2 intrinsics or scalar.
fn main() {
    // Expose target_feature to cfg! inside the crate.
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .contains("avx2")
        {
            println!("cargo:rustc-cfg=has_avx2");
        }
    }
}
