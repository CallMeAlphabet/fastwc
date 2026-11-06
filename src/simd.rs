use crate::ws::{self, WsMode};

const LANE: usize = 32;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;
#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;

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
    (chunk.simd_ge(V::splat(0x09)) & chunk.simd_le(V::splat(0x0d))) | eq_splat(chunk, b' ')
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn count_lines_avx2(data: &[u8]) -> u64 {
    let mut lines = 0u64;
    let mut i = 0usize;
    while i + 4 * LANE <= data.len() {
        lines += bitmask(eq_splat(load_at(data, i), b'\n')).count_ones() as u64;
        lines += bitmask(eq_splat(load_at(data, i + LANE), b'\n')).count_ones() as u64;
        lines += bitmask(eq_splat(load_at(data, i + 2 * LANE), b'\n')).count_ones() as u64;
        lines += bitmask(eq_splat(load_at(data, i + 3 * LANE), b'\n')).count_ones() as u64;
        i += 4 * LANE;
    }
    while i + LANE <= data.len() {
        lines += bitmask(eq_splat(load_at(data, i), b'\n')).count_ones() as u64;
        i += LANE;
    }
    lines + data[i..].iter().filter(|&&b| b == b'\n').count() as u64
}

pub fn count_lines(data: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if avx2_available() {
        return unsafe { count_lines_avx2(data) };
    }
    data.iter().filter(|&&b| b == b'\n').count() as u64
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn find_line_control_avx2(data: &[u8], start: usize) -> usize {
    let mut i = start;
    while i + LANE <= data.len() {
        let chunk = load_at(data, i);
        let mask = eq_splat(chunk, b'\n') | eq_splat(chunk, b'\t') | eq_splat(chunk, b'\r') | eq_splat(chunk, 0x0c);
        let bits = bitmask(mask);
        if bits != 0 {
            return i + bits.trailing_zeros() as usize;
        }
        i += LANE;
    }
    while i < data.len() {
        if matches!(data[i], b'\n' | b'\t' | b'\r' | 0x0c) { break; }
        i += 1;
    }
    i
}

pub fn find_line_control(data: &[u8], start: usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if avx2_available() {
        return unsafe { find_line_control_avx2(data, start) };
    }
    let mut i = start;
    while i < data.len() {
        if matches!(data[i], b'\n' | b'\t' | b'\r' | 0x0c) { break; }
        i += 1;
    }
    i
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn segment_width_avx2(data: &[u8], unicode: bool) -> i64 {
    let mut width = 0i64;
    let mut i = 0usize;
    while i + LANE + 5 <= data.len() {
        let chunk = load_at(data, i);
        let printable = chunk.simd_ge(V::splat(0x20)) & chunk.simd_le(V::splat(0x7e));
        width += bitmask(printable).count_ones() as i64;
        if unicode {
            let mut high = arch::_mm256_movemask_epi8(chunk.0) as u32 as u64;
            while high != 0 {
                let pos = high.trailing_zeros() as usize;
                let (cp, len, valid) = ws::decode(data, i + pos);
                if valid {
                    width += ws::display_width(cp, true);
                }
                high &= !(((1u64 << len) - 1) << pos);
            }
        }
        i += LANE;
    }
    while i < data.len() {
        let b = data[i];
        if (0x20..=0x7e).contains(&b) {
            width += 1;
            i += 1;
        } else if b >= 0x80 && unicode {
            let (cp, len, valid) = ws::decode(data, i);
            if valid {
                width += ws::display_width(cp, true);
            }
            i += len;
        } else {
            i += 1;
        }
    }
    width
}

pub fn segment_width(data: &[u8], unicode: bool) -> i64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if avx2_available() {
        return unsafe { segment_width_avx2(data, unicode) };
    }
    let mut width = 0i64;
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        if (0x20..=0x7e).contains(&b) {
            width += 1;
            i += 1;
        } else if b >= 0x80 && unicode {
            let (cp, len, valid) = ws::decode(data, i);
            if valid { width += ws::display_width(cp, true); }
            i += len;
        } else {
            i += 1;
        }
    }
    width
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

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn count_words_impl(data: &[u8], carry_in: bool, mode: WsMode) -> (u64, u64, bool) {
    let mut words = 0u64;
    let mut carry = carry_in;
    let mut i = 0usize;
    if !mode.unicode {
        if mode.nbsp {
            while i + LANE <= data.len() {
                let chunk = load_at(data, i);
                let ws_bits = bitmask(ascii_ws_mask(chunk) | eq_splat(chunk, ws::NBSP_BYTE));
                words += (!ws_bits & ((ws_bits << 1) | (carry as u32))).count_ones() as u64;
                carry = (ws_bits >> (LANE - 1)) & 1 == 1;
                i += LANE;
            }
        } else {
            while i + LANE <= data.len() {
                let ws_bits = bitmask(ascii_ws_mask(load_at(data, i)));
                words += (!ws_bits & ((ws_bits << 1) | (carry as u32))).count_ones() as u64;
                carry = (ws_bits >> (LANE - 1)) & 1 == 1;
                i += LANE;
            }
        }
        let (_, tail_words, _, _, tail_carry) = if mode.nbsp {
            count_unibyte_nbsp_scalar(&data[i..], carry)
        } else {
            count_buf_scalar(&data[i..], carry, false)
        };
        return (words + tail_words, data.len() as u64, tail_carry);
    }
    let mut ws_carry = 0u32;
    while i + LANE + 3 <= data.len() {
        let chunk = load_at(data, i);
        let ascii_ws_bits = bitmask(ascii_ws_mask(chunk));
        let (ws2, ws3) = ws_seq_masks_impl(data, i, mode.nbsp);
        let wide = ((ws2 as u64) | ((ws2 as u64) << 1))
            | ((ws3 as u64) | ((ws3 as u64) << 1) | ((ws3 as u64) << 2));
        let ws_all = ascii_ws_bits | (wide as u32) | ws_carry;
        ws_carry = (wide >> LANE) as u32;
        words += (!ws_all & ((ws_all << 1) | (carry as u32))).count_ones() as u64;
        carry = (ws_all >> (LANE - 1)) & 1 == 1;
        i += LANE;
    }
    i += ws_carry.count_ones() as usize;
    let (_, tail_words, _, _, tail_carry) = ws::count_scalar_unicode(&data[i..], carry, false, mode);
    (words + tail_words, data.len() as u64, tail_carry)
}

pub fn count_words_mode(data: &[u8], carry_in: bool, mode: WsMode) -> (u64, u64, bool) {
    if avx2_available() {
        return unsafe { count_words_impl(data, carry_in, mode) };
    }
    let (_, words, bytes, _, carry) = count_buf_mode(data, carry_in, false, mode);
    (words, bytes, carry)
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

    let t = next2 ^ high;
    let mut low = t.simd_le(V::splat(0x0A));
    if !nbsp {
        low &= !t.simd_eq(V::splat(0x07));
    }
    let mut sep = (t & V::splat(0xfe)).simd_eq(V::splat(0x28));
    if nbsp {
        sep |= t.simd_eq(V::splat(0x2F));
    }
    let mut narrow = t.simd_eq(V::splat(0x1F));
    if nbsp {
        narrow |= t.simd_eq(V::splat(0x20));
    }
    let e2_hit = (at1_80 & (low | sep)) | (at1_81 & narrow);
    let hit = (is_e2 & e2_hit) | (at2_80 & ((is_e1 & at1_9a) | (is_e3 & at1_80)));
    (ws2, bitmask(hit))
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "x86"), target_feature(enable = "avx2"))]
fn count_valid_utf8_chars_impl(data: &[u8]) -> u64 {
    let mut chars = 0u64;
    let mut i = 0usize;
    let cont_mask = V::splat(0xC0);
    let cont_tag = V::splat(0x80);
    while i + LANE <= data.len() {
        let chunk = load_at(data, i);
        let is_cont = (chunk & cont_mask).simd_eq(cont_tag);
        chars += (LANE as u32 - bitmask(is_cont).count_ones()) as u64;
        i += LANE;
    }
    chars + data[i..].iter().filter(|&&b| b & 0xC0 != 0x80).count() as u64
}

fn count_valid_utf8_chars(data: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if avx2_available() {
        return unsafe { count_valid_utf8_chars_impl(data) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        count_valid_utf8_chars_impl(data)
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        data.iter().filter(|&&b| b & 0xC0 != 0x80).count() as u64
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn count_unicode_chars_avx2(data: &[u8]) -> u64 {
    let mut chars = 0u64;
    let mut i = 0usize;
    let cont_mask = V::splat(0xC0);
    let cont_tag = V::splat(0x80);
    while i + LANE + 5 <= data.len() {
        let chunk = load_at(data, i);
        let next1 = V::from_slice(&data[i + 1..i + 1 + LANE]);
        let next2 = V::from_slice(&data[i + 2..i + 2 + LANE]);
        let next3 = V::from_slice(&data[i + 3..i + 3 + LANE]);
        let next4 = V::from_slice(&data[i + 4..i + 4 + LANE]);
        let next5 = V::from_slice(&data[i + 5..i + 5 + LANE]);
        let cont1 = bitmask((next1 & cont_mask).simd_eq(cont_tag));
        let cont2 = bitmask((next2 & cont_mask).simd_eq(cont_tag));
        let cont3 = bitmask((next3 & cont_mask).simd_eq(cont_tag));
        let cont4 = bitmask((next4 & cont_mask).simd_eq(cont_tag));
        let cont5 = bitmask((next5 & cont_mask).simd_eq(cont_tag));
        let range = |v: V, lo: u8, hi: u8| bitmask(v.simd_ge(V::splat(lo)) & v.simd_le(V::splat(hi)));

        let lead2 = range(chunk, 0xC2, 0xDF);
        let lead3 = range(chunk, 0xE0, 0xEF);
        let lead4 = range(chunk, 0xF0, 0xF7);
        let lead5 = range(chunk, 0xF8, 0xFB);
        let lead6 = range(chunk, 0xFC, 0xFD);
        let valid2 = lead2 & cont1;

        let e0 = bitmask(eq_splat(chunk, 0xE0));
        let ed = bitmask(eq_splat(chunk, 0xED));
        let valid3_start = (lead3 & !e0 & !ed)
            | (e0 & range(next1, 0xA0, 0xBF))
            | (ed & range(next1, 0x80, 0x9F));
        let valid3 = valid3_start & cont1 & cont2;

        let f0 = bitmask(eq_splat(chunk, 0xF0));
        let valid4_start = (lead4 & !f0) | (f0 & range(next1, 0x90, 0xBF));
        let valid4 = valid4_start & cont1 & cont2 & cont3;

        let f8 = bitmask(eq_splat(chunk, 0xF8));
        let valid5_start = (lead5 & !f8) | (f8 & range(next1, 0x88, 0xBF));
        let valid5 = valid5_start & cont1 & cont2 & cont3 & cont4;

        let fc = bitmask(eq_splat(chunk, 0xFC));
        let valid6_start = (lead6 & !fc) | (fc & range(next1, 0x84, 0xBF));
        let valid6 = valid6_start & cont1 & cont2 & cont3 & cont4 & cont5;

        let high = arch::_mm256_movemask_epi8(chunk.0) as u32;
        chars += (LANE as u32 - high.count_ones()) as u64;
        chars += valid2.count_ones() as u64;
        chars += valid3.count_ones() as u64;
        chars += valid4.count_ones() as u64;
        chars += valid5.count_ones() as u64;
        chars += valid6.count_ones() as u64;
        i += LANE;
    }
    chars + count_unicode_chars_scalar(&data[i..])
}

fn count_unicode_chars(data: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    if avx2_available() {
        return unsafe { count_unicode_chars_avx2(data) };
    }
    count_unicode_chars_scalar(data)
}

fn count_unicode_chars_scalar(data: &[u8]) -> u64 {
    let mut chars = 0u64;
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        if b < 0x80 {
            chars += 1;
            i += 1;
        } else {
            let (_, len, valid) = ws::decode(data, i);
            chars += valid as u64;
            i += len;
        }
    }
    chars
}

pub fn count_chars_only(data: &[u8], mode: WsMode) -> u64 {
    if !mode.unicode || !data.iter().any(|&b| b >= 0x80) {
        return data.len() as u64;
    }
    if std::str::from_utf8(data).is_ok() {
        return count_valid_utf8_chars(data);
    }
    count_unicode_chars(data)
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

    if want_chars {
        if !data.iter().any(|&b| b >= 0x80) {
            return count_buf_portable_impl(data, carry_in, true);
        }
        let valid_utf8 = std::str::from_utf8(data).is_ok();
        let (lines, words, bytes, _, carry) = count_buf_unicode_impl(data, carry_in, false, mode);
        let chars = if valid_utf8 { count_valid_utf8_chars(data) } else { count_unicode_chars(data) };
        return (lines, words, bytes, chars, carry);
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
