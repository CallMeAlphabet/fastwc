//! Copyright 2026 CallMeAlphabet (ItzAlphabet)
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!    http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

//! Selects the counting kernel at build time.
//!
//! `std::simd` (the `portable_simd` feature) is still nightly-only, so the
//! vectorised kernel in `src/simd.rs` can only be compiled by a compiler that
//! accepts the feature gate. This script probes the active `rustc` and sets
//! `cfg(simd_portable)` when it does; `src/simd_scalar.rs` is used otherwise,
//! which is what lets `cargo install fastwc` work on a stable toolchain.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(simd_portable)");

    if portable_simd_supported() {
        println!("cargo:rustc-cfg=simd_portable");
    }
}

/// True when `rustc` is a nightly/dev toolchain, i.e. one that accepts
/// `#![feature(portable_simd)]`.
fn portable_simd_supported() -> bool {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());

    let Ok(out) = Command::new(rustc).arg("-vV").output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }

    // `rustc -vV` prints one `key: value` line per field; the one we want is
    // `release: 1.100.0-nightly` (nightly), `release: 1.95.0` (stable) or
    // `release: 1.96.0-dev` (bors / `rustup toolchain link` builds).
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .is_some_and(|release| release.contains("nightly") || release.contains("-dev"))
}
