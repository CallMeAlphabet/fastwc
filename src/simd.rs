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

//! Portable SIMD counting kernel for fastwc.
//! LLVM lowers `Simd<u8, 32>` to AVX2 on x86_64 and NEON (2×16) on aarch64.

use crate::ws::{self, WsMode};

const LANE: usize = 32;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;
#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;

/// One 32-byte lane and the mask type it compares into.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod lane {
    use super::arch::*;

    #[derive(Clone, Copy)]
    pub struct V(pub __m256i);
    #[derive(Clone, Copy)]
    pub struct M(pub __m256i);

    impl V {
        #[inline(always)]
        pub fn splat(b: u8) -> V {
            unsafe { V(_mm256_set1_epi8(b as i8)) }
        }
        #[inline(always)]
        pub fn from_slice(s: &[u8]) -> V {
            unsafe { V(_mm256_loadu_si256(s.as_ptr() as *const __m256i)) }
        }
        #[inline(always)]
        pub fn simd_eq(self, o: V) -> M {
            unsafe { M(_mm256_cmpeq_epi8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn simd_le(self, o: V) -> M {
            unsafe { M(_mm256_cmpeq_epi8(_mm256_min_epu8(self.0, o.0), self.0)) }
        }
        #[inline(always)]
        pub fn simd_ge(self, o: V) -> M {
            unsafe { M(_mm256_cmpeq_epi8(_mm256_max_epu8(self.0, o.0), self.0)) }
        }
    }
    impl std::ops::BitAnd for V {
        type Output = V;
        #[inline(always)]
        fn bitand(self, o: V) -> V {
            unsafe { V(_mm256_and_si256(self.0, o.0)) }
        }
    }
    impl std::ops::BitXor for V {
        type Output = V;
        #[inline(always)]
        fn bitxor(self, o: V) -> V {
            unsafe { V(_mm256_xor_si256(self.0, o.0)) }
        }
    }
    impl M {
        #[inline(always)]
        pub fn splat(b: bool) -> M {
            unsafe { M(_mm256_set1_epi8(if b { -1 } else { 0 })) }
        }
        #[inline(always)]
        pub fn to_bitmask(self) -> u32 {
            unsafe { _mm256_movemask_epi8(self.0) as u32 }
        }
    }
    impl std::ops::BitAnd for M {
        type Output = M;
        #[inline(always)]
        fn bitand(self, o: M) -> M {
            unsafe { M(_mm256_and_si256(self.0, o.0)) }
        }
    }
    impl std::ops::BitAndAssign for M {
        #[inline(always)]
        fn bitand_assign(&mut self, o: M) {
            *self = *self & o;
        }
    }
    impl std::ops::BitOr for M {
        type Output = M;
        #[inline(always)]
        fn bitor(self, o: M) -> M {
            unsafe { M(_mm256_or_si256(self.0, o.0)) }
        }
    }
    impl std::ops::BitOrAssign for M {
        #[inline(always)]
        fn bitor_assign(&mut self, o: M) {
            *self = *self | o;
        }
    }
    impl std::ops::Not for M {
        type Output = M;
        #[inline(always)]
        fn not(self) -> M {
            unsafe { M(_mm256_xor_si256(self.0, _mm256_set1_epi8(-1))) }
        }
    }
}

/// Same lane API on plain arrays for every other architecture; the compiler
/// auto-vectorises these loops with whatever the target offers.
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
mod lane {
    use super::LANE;

    #[derive(Clone, Copy)]
    pub struct V(pub [u8; LANE]);
    #[derive(Clone, Copy)]
    pub struct M(pub [bool; LANE]);

    impl V {
        #[inline(always)]
        pub fn splat(b: u8) -> V {
            V([b; LANE])
        }
        #[inline(always)]
        pub fn from_slice(s: &[u8]) -> V {
            V(s[..LANE].try_into().unwrap())
        }
        #[inline(always)]
        pub fn simd_eq(self, o: V) -> M {
            M(std::array::from_fn(|i| self.0[i] == o.0[i]))
        }
        #[inline(always)]
        pub fn simd_le(self, o: V) -> M {
            M(std::array::from_fn(|i| self.0[i] <= o.0[i]))
        }
        #[inline(always)]
        pub fn simd_ge(self, o: V) -> M {
            M(std::array::from_fn(|i| self.0[i] >= o.0[i]))
        }
    }
    impl std::ops::BitAnd for V {
        type Output = V;
        #[inline(always)]
        fn bitand(self, o: V) -> V {
            V(std::array::from_fn(|i| self.0[i] & o.0[i]))
        }
    }
    impl std::ops::BitXor for V {
        type Output = V;
        #[inline(always)]
        fn bitxor(self, o: V) -> V {
            V(std::array::from_fn(|i| self.0[i] ^ o.0[i]))
        }
    }
    impl M {
        #[inline(always)]
        pub fn splat(b: bool) -> M {
            M([b; LANE])
        }
        #[inline(always)]
        pub fn to_bitmask(self) -> u32 {
            let mut m = 0u32;
            for i in 0..LANE {
                m |= (self.0[i] as u32) << i;
            }
            m
        }
    }
    impl std::ops::BitAnd for M {
        type Output = M;
        #[inline(always)]
        fn bitand(self, o: M) -> M {
            M(std::array::from_fn(|i| self.0[i] & o.0[i]))
        }
    }
    impl std::ops::BitAndAssign for M {
        #[inline(always)]
        fn bitand_assign(&mut self, o: M) {
            *self = *self & o;
        }
    }
    impl std::ops::BitOr for M {
        type Output = M;
        #[inline(always)]
        fn bitor(self, o: M) -> M {
            M(std::array::from_fn(|i| self.0[i] | o.0[i]))
        }
    }
    impl std::ops::BitOrAssign for M {
        #[inline(always)]
        fn bitor_assign(&mut self, o: M) {
            *self = *self | o;
        }
    }
    impl std::ops::Not for M {
        type Output = M;
        #[inline(always)]
        fn not(self) -> M {
            M(std::array::from_fn(|i| !self.0[i]))
        }
    }
}

use lane::{M, V};

/// Kept for debug output / ABI compatibility; true when the AVX2 lane code
/// can run on this CPU. Every entry point is dispatched on the same check.
pub fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        true
    }
}

#[inline(always)]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

#[inline(always)]
fn load_at(data: &[u8], i: usize) -> V {
    V::from_slice(&data[i..i + LANE])
}

#[inline(always)]
fn bitmask(m: M) -> u32 {
    m.to_bitmask()
}

#[inline(always)]
fn eq_splat(chunk: V, b: u8) -> M {
    chunk.simd_eq(V::splat(b))
}

#[inline(always)]
fn ascii_ws_mask(chunk: V) -> M {
    eq_splat(chunk, b'\n')
        | eq_splat(chunk, b' ')
        | eq_splat(chunk, b'\t')
        | eq_splat(chunk, 0x0b)
        | eq_splat(chunk, 0x0c)
        | eq_splat(chunk, b'\r')
}

fn count_buf_portable(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
    if avx2_available() {
        return unsafe { count_buf_portable_impl(data, carry_in, want_chars) };
    }
    count_buf_scalar(data, carry_in, want_chars)
}

fn count_buf_unibyte_nbsp(data: &[u8], carry_in: bool) -> (u64, u64, u64, u64, bool) {
    if avx2_available() {
        return unsafe { count_buf_unibyte_nbsp_impl(data, carry_in) };
    }
    count_unibyte_nbsp_scalar(data, carry_in)
}

fn count_buf_unicode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    if avx2_available() {
        return unsafe { count_buf_unicode_impl(data, carry_in, want_chars, mode) };
    }
    ws::count_scalar_unicode(data, carry_in, want_chars, mode)
}

pub fn simple_ascii_run(data: &[u8], start: usize, carry_ws: bool, avx2: bool) -> (usize, u64, bool) {
    if avx2 {
        return unsafe { simple_ascii_run_impl(data, start, carry_ws) };
    }
    simple_ascii_run_scalar(data, start, carry_ws)
}

pub fn count_buf(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
    count_buf_portable(data, carry_in, want_chars)
}

pub fn count_buf_mode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    if !mode.unicode {
        let (lines, words, bytes, _, carry) = if mode.nbsp {
            count_buf_unibyte_nbsp(data, carry_in)
        } else {
            count_buf(data, carry_in, false)
        };
        return (lines, words, bytes, if want_chars { bytes } else { 0 }, carry);
    }
    count_buf_unicode(data, carry_in, want_chars, mode)
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn count_buf_unibyte_nbsp_impl(data: &[u8], carry_in: bool) -> (u64, u64, u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut carry = carry_in;
    let mut i = 0usize;
    while i + LANE <= data.len() {
        let chunk = load_at(data, i);
        let eq_nl = eq_splat(chunk, b'\n');
        let ws = ascii_ws_mask(chunk) | eq_splat(chunk, ws::NBSP_BYTE);
        let nl_bits = bitmask(eq_nl);
        let ws_bits = bitmask(ws);
        lines += nl_bits.count_ones() as u64;
        words += (!ws_bits & ((ws_bits << 1) | (carry as u32))).count_ones() as u64;
        carry = (ws_bits >> (LANE - 1)) & 1 == 1;
        i += LANE;
    }
    let (t_lines, t_words, _, _, t_carry) = count_unibyte_nbsp_scalar(&data[i..], carry);
    (lines + t_lines, words + t_words, data.len() as u64, 0, t_carry)
}

fn count_unibyte_nbsp_scalar(data: &[u8], carry_in: bool) -> (u64, u64, u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut prev_ws = carry_in;
    for &b in data {
        if b == b'\n' {
            lines += 1;
        }
        let ws = is_ws_byte(b) || b == ws::NBSP_BYTE;
        if !ws && prev_ws {
            words += 1;
        }
        prev_ws = ws;
    }
    (lines, words, data.len() as u64, 0, prev_ws)
}

#[inline]
fn count_buf_scalar(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let mut prev_ws = carry_in;
    for &b in data {
        if b == b'\n' {
            lines += 1;
        }
        let ws = is_ws_byte(b);
        if !ws && prev_ws {
            words += 1;
        }
        prev_ws = ws;
        if want_chars && (b & 0xC0) != 0x80 {
            chars += 1;
        }
    }
    (lines, words, data.len() as u64, chars, prev_ws)
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn count_buf_portable_impl(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let mut carry = carry_in;
    let mut i = 0usize;
    let cont_mask = V::splat(0xC0);
    let cont_tag = V::splat(0x80);
    while i + LANE <= data.len() {
        let chunk = load_at(data, i);
        let eq_nl = eq_splat(chunk, b'\n');
        let ws = ascii_ws_mask(chunk);
        let nl_bits = bitmask(eq_nl);
        let ws_bits = bitmask(ws);
        lines += nl_bits.count_ones() as u64;
        words += (!ws_bits & ((ws_bits << 1) | (carry as u32))).count_ones() as u64;
        carry = (ws_bits >> (LANE - 1)) & 1 == 1;
        if want_chars {
            let is_cont = (chunk & cont_mask).simd_eq(cont_tag);
            chars += (LANE as u32 - bitmask(is_cont).count_ones()) as u64;
        }
        i += LANE;
    }
    let (t_lines, t_words, _, t_chars, t_carry) = count_buf_scalar(&data[i..], carry, want_chars);
    (lines + t_lines, words + t_words, data.len() as u64, chars + t_chars, t_carry)
}

/// Positions of 2-byte and 3-byte whitespace leads in one 32-byte lane.
#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn ws_seq_masks_impl(data: &[u8], i: usize, nbsp: bool) -> (u32, u32) {
    let chunk = load_at(data, i);
    let next1 = V::from_slice(&data[i + 1..i + 1 + LANE]);
    let next2 = V::from_slice(&data[i + 2..i + 2 + LANE]);
    let high = V::splat(0x80);

    let is_c2 = eq_splat(chunk, ws::LEAD_C2);
    let is_e1 = eq_splat(chunk, ws::LEAD_E1);
    let is_e2 = eq_splat(chunk, ws::LEAD_E2);
    let is_e3 = eq_splat(chunk, ws::LEAD_E3);

    let cand = is_e1 | is_e2 | is_e3 | if nbsp { is_c2 } else { M::splat(false) };
    if bitmask(cand) == 0 {
        return (0, 0);
    }

    let at1_80 = next1.simd_eq(high);
    let at2_80 = next2.simd_eq(high);
    let at1_81 = eq_splat(next1, 0x81);
    let at1_9a = eq_splat(next1, 0x9a);

    let ws2 = if nbsp {
        bitmask(is_c2 & eq_splat(next1, 0xa0))
    } else {
        0
    };

    // U+2000..U+2006, U+2008..U+200A (and U+2007 if GNU)
    // After xor 0x80 the low separators' final bytes 0x80..=0x8A land in 0x00..=0x0A.
    let t = next2 ^ high;
    let mut low = t.simd_le(V::splat(0x0A));
    if !nbsp {
        low &= !t.simd_eq(V::splat(0x07)); // U+2007
    }
    let mut sep = (t & V::splat(0xfe)).simd_eq(V::splat(0x28)); // 0xa8 ^ 0x80 = 0x28 (U+2028/2029)
    if nbsp {
        sep |= t.simd_eq(V::splat(0x2F)); // 0xaf ^ 0x80
    }
    let mut narrow = t.simd_eq(V::splat(0x1F)); // 0x9f ^ 0x80 U+205F
    if nbsp {
        narrow |= t.simd_eq(V::splat(0x20)); // U+2060
    }
    let e2_hit = (at1_80 & (low | sep)) | (at1_81 & narrow);
    let hit = (is_e2 & e2_hit) | (at2_80 & ((is_e1 & at1_9a) | (is_e3 & at1_80)));
    (ws2, bitmask(hit))
}

pub fn count_chars_only(data: &[u8], mode: WsMode) -> u64 {
    if !mode.unicode {
        return data.len() as u64;
    }
    ws::count_scalar_unicode(data, true, true, mode).3
}

#[cfg(test)]
fn chars_tail_scalar(data: &[u8], from: usize) -> u64 {
    let mut n = 0u64;
    for i in from..data.len() {
        let b = data[i];
        if b < 0x80 {
            n += 1;
        } else if ws::decode(data, i).2 {
            n += 1;
        }
    }
    n
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn count_buf_unicode_impl(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    // Only -m needs every sequence decoded; use the scalar decoder for that
    // (still vectorised via count_buf_portable for ASCII-only).
    if want_chars {
        if data.iter().any(|&b| b >= 0x80) {
            return ws::count_scalar_unicode(data, carry_in, want_chars, mode);
        }
        return count_buf_portable_impl(data, carry_in, true);
    }
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut carry = carry_in;
    let mut ws_carry = 0u32;
    let mut i = 0usize;
    while i + LANE + 3 <= data.len() {
        let chunk = load_at(data, i);
        let eq_nl = eq_splat(chunk, b'\n');
        let ascii_ws_bits = bitmask(ascii_ws_mask(chunk));
        let nl_bits = bitmask(eq_nl);

        let (ws2, ws3) = ws_seq_masks_impl(data, i, mode.nbsp);
        let wide = ((ws2 as u64) | ((ws2 as u64) << 1))
            | ((ws3 as u64) | ((ws3 as u64) << 1) | ((ws3 as u64) << 2));
        let ws_all = ascii_ws_bits | (wide as u32) | ws_carry;
        ws_carry = (wide >> LANE) as u32;

        lines += nl_bits.count_ones() as u64;
        words += (!ws_all & ((ws_all << 1) | (carry as u32))).count_ones() as u64;
        carry = (ws_all >> (LANE - 1)) & 1 == 1;
        i += LANE;
    }
    i += ws_carry.count_ones() as usize;
    let (t_lines, t_words, _, _, t_carry) =
        ws::count_scalar_unicode(&data[i..], carry, false, mode);
    (lines + t_lines, words + t_words, data.len() as u64, 0, t_carry)
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn simple_ascii_run_impl(data: &[u8], start: usize, carry_ws: bool) -> (usize, u64, bool) {
    let mut i = start;
    let mut words = 0u64;
    let mut carry = carry_ws;
    let v20 = V::splat(0x20);
    let v7e = V::splat(0x7e);
    let space = V::splat(b' ');
    while i + LANE <= data.len() {
        let chunk = load_at(data, i);
        let simple = chunk.simd_ge(v20) & chunk.simd_le(v7e);
        let bad = !bitmask(simple);
        let sp = bitmask(eq_splat(chunk, b' '));
        let n = if bad == 0 { LANE } else { bad.trailing_zeros() as usize };
        if n > 0 {
            let mask = if n == LANE { u32::MAX } else { (1u32 << n) - 1 };
            let sp = sp & mask;
            let prev = (sp << 1) | (carry as u32);
            words += (!sp & prev & mask).count_ones() as u64;
            carry = (sp >> (n - 1)) & 1 == 1;
        }
        i += n;
        if bad != 0 {
            return (i, words, carry);
        }
        let _ = (space, v20, v7e);
    }
    let (end, w, carry) = simple_ascii_run_scalar(data, i, carry);
    (end, words + w, carry)
}

fn simple_ascii_run_scalar(data: &[u8], start: usize, carry_ws: bool) -> (usize, u64, bool) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
        count_buf_scalar(data, carry_in, want_chars)
    }

    #[test]
    fn portable_matches_scalar_on_ascii() {
        let data: Vec<u8> = b"the quick brown fox\tjumps\nover the lazy dog  again and again "
            .iter()
            .cycle()
            .take(5000)
            .copied()
            .collect();
        for carry_in in [true, false] {
            let a = count_buf_portable(&data, carry_in, true);
            let b = reference(&data, carry_in, true);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn basic_counts() {
        let (lines, words, bytes, _chars, carry) =
            count_buf_scalar(b"hello world\nfoo  bar\n", true, false);
        assert_eq!(lines, 2);
        assert_eq!(words, 4);
        assert_eq!(bytes, 21);
        assert_eq!(carry, true);
    }

    #[test]
    fn unicode_kernel_matches_scalar() {
        let mode = WsMode { unicode: true, nbsp: true };
        let pieces: &[&[u8]] = &[
            b"a", b"b", b" ", b"\t", b"\n",
            "\u{00A0}".as_bytes(), "\u{2003}".as_bytes(), "\u{3000}".as_bytes(),
            "\u{1680}".as_bytes(), b"word",
        ];
        let mut data = Vec::new();
        for p in pieces.iter().cycle().take(200) {
            data.extend_from_slice(p);
        }
        let expected = ws::count_scalar_unicode(&data, true, false, mode);
        let actual = count_buf_mode(&data, true, false, mode);
        assert_eq!((expected.0, expected.1, expected.4), (actual.0, actual.1, actual.4));
    }

    #[test]
    fn _chars_tail_used() {
        let _ = chars_tail_scalar(b"abc", 0);
    }
}
