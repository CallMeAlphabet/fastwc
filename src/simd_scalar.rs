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

//! Stable-toolchain counting kernel.
//!
//! Same entry points and same results as the portable SIMD kernel in
//! `simd.rs`, minus the vectors: this is the fallback used whenever the
//! compiler does not accept `#![feature(portable_simd)]`. The counting
//! rules are the ones `simd.rs` was verified against, so the two kernels
//! are interchangeable and the test suite runs unchanged on both.

use crate::ws::{self, WsMode};

/// Always false: this build has no vectorised path to report.
pub fn avx2_available() -> bool {
    false
}

#[inline(always)]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Unibyte locale: lines, words and the trailing whitespace carry.
fn count_buf_unibyte(data: &[u8], carry_in: bool, nbsp: bool) -> (u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut prev_ws = carry_in;
    for &b in data {
        if b == b'\n' {
            lines += 1;
        }
        let ws = is_ws_byte(b) || (nbsp && b == ws::NBSP_BYTE);
        if !ws && prev_ws {
            words += 1;
        }
        prev_ws = ws;
    }
    (lines, words, prev_ws)
}

pub fn count_buf_mode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    if mode.unicode {
        return ws::count_scalar_unicode(data, carry_in, want_chars, mode);
    }
    // In a unibyte locale a character is a byte, so chars == bytes.
    let (lines, words, carry) = count_buf_unibyte(data, carry_in, mode.nbsp);
    let bytes = data.len() as u64;
    (
        lines,
        words,
        bytes,
        if want_chars { bytes } else { 0 },
        carry,
    )
}

pub fn count_chars_only(data: &[u8], mode: WsMode) -> u64 {
    if !mode.unicode {
        return data.len() as u64;
    }
    ws::count_scalar_unicode(data, true, true, mode).3
}

/// Scan the run of printable-ASCII bytes starting at `start`, counting the
/// words it closes. Returns (end, words, carry_ws).
pub fn simple_ascii_run(
    data: &[u8],
    start: usize,
    carry_ws: bool,
    _avx2: bool,
) -> (usize, u64, bool) {
    let mut i = start;
    let mut words = 0u64;
    let mut carry = carry_ws;
    while i < data.len() {
        let b = data[i];
        if !(0x20..=0x7e).contains(&b) {
            break;
        }
        let ws = b == b' ';
        if !ws && carry {
            words += 1;
        }
        carry = ws;
        i += 1;
    }
    (i, words, carry)
}
