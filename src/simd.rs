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

//! SIMD-accelerated counting kernel for fastwc.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ws::{self, WsMode};

pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

pub fn count_buf(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_buf_avx2(data, carry_in, want_chars) };
        }
    }
    count_buf_scalar(data, carry_in, want_chars)
}

/// Locale-aware entry point.
///
/// In a multibyte locale we still run the AVX2 kernel, because
/// non-ASCII whitespace can only begin with one of four lead bytes (0xC2, 0xE1,
/// 0xE2, 0xE3). Each lane is checked for those bytes with a few extra compares;
/// lanes that contain none are counted by the plain ASCII logic, which is the
/// overwhelmingly common case. Only when a candidate lead byte shows up do we
/// fall back to decoding, and then only for the affected region.
pub fn count_buf_mode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    // Unibyte locale: every byte is one character, so `-m` equals `-c` and no
    // sequence is ever decoded. Byte 0xA0 is still a delimiter unless
    // POSIXLY_CORRECT, because GNU builds its table through ISO-8859-1.
    if !mode.unicode {
        let (lines, words, bytes, _, carry) = if mode.nbsp {
            count_buf_unibyte_nbsp(data, carry_in)
        } else {
            count_buf(data, carry_in, false)
        };
        return (lines, words, bytes, if want_chars { bytes } else { 0 }, carry);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !want_chars
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
        {
            return unsafe { count_lw_avx512(data, carry_in, mode) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_buf_avx2_unicode(data, carry_in, want_chars, mode) };
        }
    }

    ws::count_scalar_unicode(data, carry_in, want_chars, mode)
}

/// Unibyte locale with GNU's non-breaking-space extension: ASCII whitespace
/// plus byte 0xA0.
fn count_buf_unibyte_nbsp(data: &[u8], carry_in: bool) -> (u64, u64, u64, u64, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_buf_avx2_unibyte_nbsp(data, carry_in) };
        }
    }
    count_unibyte_nbsp_scalar(data, carry_in)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_buf_avx2_unibyte_nbsp(data: &[u8], carry_in: bool) -> (u64, u64, u64, u64, bool) {
    const LANE: usize = 32;

    let mut lines = 0u64;
    let mut words = 0u64;
    let mut carry = carry_in;

    let newline = _mm256_set1_epi8(b'\n' as i8);
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let vtab = _mm256_set1_epi8(0x0bi8);
    let ff = _mm256_set1_epi8(0x0ci8);
    let cr = _mm256_set1_epi8(b'\r' as i8);
    let nbsp = _mm256_set1_epi8(ws::NBSP_BYTE as i8);

    let mut i = 0usize;
    while i + LANE <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

        let eq_nl = _mm256_cmpeq_epi8(chunk, newline);
        let ws_vec = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(eq_nl, _mm256_cmpeq_epi8(chunk, space)),
                _mm256_or_si256(_mm256_cmpeq_epi8(chunk, tab), _mm256_cmpeq_epi8(chunk, vtab)),
            ),
            _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(chunk, ff), _mm256_cmpeq_epi8(chunk, cr)),
                _mm256_cmpeq_epi8(chunk, nbsp),
            ),
        );

        let nl_bits = _mm256_movemask_epi8(eq_nl) as u32;
        let ws_bits = _mm256_movemask_epi8(ws_vec) as u32;

        lines += nl_bits.count_ones() as u64;
        words += (!ws_bits & ((ws_bits << 1) | (carry as u32))).count_ones() as u64;
        carry = (ws_bits >> (LANE - 1)) & 1 == 1;

        i += LANE;
    }

    let (t_lines, t_words, _, _, t_carry) = count_unibyte_nbsp_scalar(&data[i..], carry);
    (lines + t_lines, words + t_words, data.len() as u64, 0, t_carry)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_buf_avx2(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
) -> (u64, u64, u64, u64, bool) {
    const LANE: usize = 32;

    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let mut carry = carry_in;

    let newline = _mm256_set1_epi8(b'\n' as i8);
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let vtab = _mm256_set1_epi8(0x0bi8);
    let ff = _mm256_set1_epi8(0x0ci8);
    let cr = _mm256_set1_epi8(b'\r' as i8);
    let cont_mask = _mm256_set1_epi8(0xC0u8 as i8);
    let cont_tag = _mm256_set1_epi8(0x80u8 as i8);

    let mut i = 0usize;
    while i + LANE <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

        let eq_nl = _mm256_cmpeq_epi8(chunk, newline);
        let eq_sp = _mm256_cmpeq_epi8(chunk, space);
        let eq_tab = _mm256_cmpeq_epi8(chunk, tab);
        let eq_vt = _mm256_cmpeq_epi8(chunk, vtab);
        let eq_ff = _mm256_cmpeq_epi8(chunk, ff);
        let eq_cr = _mm256_cmpeq_epi8(chunk, cr);

        let ws_vec = _mm256_or_si256(
            _mm256_or_si256(_mm256_or_si256(eq_nl, eq_sp), _mm256_or_si256(eq_tab, eq_vt)),
            _mm256_or_si256(eq_ff, eq_cr),
        );

        let nl_bits = _mm256_movemask_epi8(eq_nl) as u32;
        let ws_bits = _mm256_movemask_epi8(ws_vec) as u32;

        lines += nl_bits.count_ones() as u64;

        let prev_ws_bits = (ws_bits << 1) | (carry as u32);
        let non_ws_bits = !ws_bits;
        let word_start_bits = non_ws_bits & prev_ws_bits;
        words += word_start_bits.count_ones() as u64;

        carry = (ws_bits >> (LANE - 1)) & 1 == 1;

        if want_chars {
            let masked = _mm256_and_si256(chunk, cont_mask);
            let is_cont = _mm256_cmpeq_epi8(masked, cont_tag);
            let cont_bits = _mm256_movemask_epi8(is_cont) as u32;
            chars += (LANE as u32 - cont_bits.count_ones()) as u64;
        }

        i += LANE;
    }

    let (t_lines, t_words, _t_bytes, t_chars, t_carry) =
        count_buf_scalar(&data[i..], carry, want_chars);

    lines += t_lines;
    words += t_words;
    chars += t_chars;

    (lines, words, data.len() as u64, chars, t_carry)
}

/// Locate multi-byte whitespace characters inside one 32-byte lane.
///
/// Returns two bitmasks of *lead* byte positions: two-byte characters (only
/// U+00A0, and only under the GNU extension) and three-byte ones. The four
/// lead bytes involved are never continuation bytes, so a match here always
/// sits on a character boundary regardless of surrounding validity.
///
/// The body is branchless and `NBSP` is a constant, so the whole thing folds
/// into a straight run of compares; text that mixes scripts would otherwise
/// mispredict on nearly every lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ws_seq_masks<const NBSP: bool>(
    chunk: __m256i,
    next1: __m256i,
    next2: __m256i,
) -> (u32, u32) {
    let high = _mm256_set1_epi8(0x80u8 as i8);

    let is_c2 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_C2 as i8));
    let is_e1 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E1 as i8));
    let is_e2 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E2 as i8));
    let is_e3 = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E3 as i8));

    let cand = _mm256_or_si256(
        _mm256_or_si256(is_e1, is_e2),
        if NBSP { _mm256_or_si256(is_e3, is_c2) } else { is_e3 },
    );
    if _mm256_movemask_epi8(cand) == 0 {
        return (0, 0);
    }

    let at1_80 = _mm256_cmpeq_epi8(next1, high);
    let at2_80 = _mm256_cmpeq_epi8(next2, high);
    let at1_81 = _mm256_cmpeq_epi8(next1, _mm256_set1_epi8(0x81u8 as i8));
    let at1_9a = _mm256_cmpeq_epi8(next1, _mm256_set1_epi8(0x9au8 as i8));

    let ws2 = if NBSP {
        let at1_a0 = _mm256_cmpeq_epi8(next1, _mm256_set1_epi8(0xa0u8 as i8));
        _mm256_movemask_epi8(_mm256_and_si256(is_c2, at1_a0)) as u32
    } else {
        0
    };

    let tail = _mm256_xor_si256(next2, high);

    // U+2000..U+2006, U+2008..U+200A, U+2028 and U+2029, plus the GNU
    // extensions U+2007 and U+202F.
    let mut low = _mm256_and_si256(
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0x0b), tail),
        _mm256_cmpgt_epi8(tail, _mm256_set1_epi8(-1)),
    );
    if !NBSP {
        low = _mm256_andnot_si256(_mm256_cmpeq_epi8(tail, _mm256_set1_epi8(0x07)), low);
    }
    let mut sep = _mm256_cmpeq_epi8(
        _mm256_and_si256(tail, _mm256_set1_epi8(0xfeu8 as i8)),
        _mm256_set1_epi8(0x28),
    );
    if NBSP {
        sep = _mm256_or_si256(sep, _mm256_cmpeq_epi8(tail, _mm256_set1_epi8(0x2f)));
    }

    // U+205F, plus the GNU extension U+2060.
    let mut narrow = _mm256_cmpeq_epi8(tail, _mm256_set1_epi8(0x1f));
    if NBSP {
        narrow = _mm256_or_si256(narrow, _mm256_cmpeq_epi8(tail, _mm256_set1_epi8(0x20)));
    }

    let e2_hit = _mm256_or_si256(
        _mm256_and_si256(at1_80, _mm256_or_si256(low, sep)),
        _mm256_and_si256(at1_81, narrow),
    );

    let hit = _mm256_or_si256(
        _mm256_and_si256(is_e2, e2_hit),
        _mm256_and_si256(
            at2_80,
            _mm256_or_si256(
                _mm256_and_si256(is_e1, at1_9a),
                _mm256_and_si256(is_e3, at1_80),
            ),
        ),
    );

    (ws2, _mm256_movemask_epi8(hit) as u32)
}

/// Character-only counting for `-m` without `-l` or `-w`.
///
/// The general kernel spends most of its work on whitespace: six compares per
/// lane, the word-transition bitmask and the carry between lanes. None of it
/// is printed when only characters are asked for, where the answer is just the
/// number of bytes that are not continuations. UTF-8 validation still has to
/// run, because a malformed sequence counts differently.
pub fn count_chars_only(data: &[u8], mode: WsMode) -> u64 {
    if !mode.unicode {
        return data.len() as u64;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { count_chars_only_avx512(data, mode) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_chars_only_avx2(data, mode) };
        }
    }
    ws::count_scalar_unicode(data, true, true, mode).3
}

/// AVX-512 form of the character counter.
///
/// Doubling the lane to 64 bytes halves the loop overhead, and the compare
/// results arrive directly in mask registers, so the `vpmovmskb` round-trip
/// through a general register disappears. On this machine it reaches memory
/// bandwidth where the AVX2 version does not.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn count_chars_only_avx512(data: &[u8], mode: WsMode) -> u64 {
    const LANE: usize = 64;
    let _ = mode;

    // Every position is judged on its own. A byte starts a character when it is
    // ASCII, or when it is a lead whose whole sequence is present and
    // well-formed. Whether some earlier byte already claimed this position does
    // not matter: the interior bytes of a sequence are all 0x80..=0xBF, and no
    // byte in that range is ever accepted as a start, so a sequence can never
    // begin inside another one. The greedy left-to-right walk and this
    // per-position test therefore always agree, which means the count carries
    // no state from lane to lane and needs no validation gate or scalar
    // fallback -- malformed input costs exactly what clean input costs.
    let mut chars = 0u64;

    let v_3f = _mm512_set1_epi8(0x3f);
    let v_c1 = _mm512_set1_epi8(0xc1u8 as i8);
    let v_fd = _mm512_set1_epi8(0xfdu8 as i8);
    let bias = _mm512_set1_epi8(0x80u8 as i8);

    // Lead-byte length thresholds, compared in the biased (signed) domain.
    let t_e0 = _mm512_set1_epi8((0xE0u8 ^ 0x80) as i8 - 1);
    let t_f0 = _mm512_set1_epi8((0xF0u8 ^ 0x80) as i8 - 1);
    let t_f8 = _mm512_set1_epi8((0xF8u8 ^ 0x80) as i8 - 1);
    let t_fc = _mm512_set1_epi8((0xFCu8 ^ 0x80) as i8 - 1);

    let e0 = _mm512_set1_epi8(0xe0u8 as i8);
    let ed = _mm512_set1_epi8(0xedu8 as i8);
    let f0 = _mm512_set1_epi8(0xf0u8 as i8);
    let f8 = _mm512_set1_epi8(0xf8u8 as i8);
    let fc = _mm512_set1_epi8(0xfcu8 as i8);

    let s_a0 = _mm512_set1_epi8((0xA0u8 ^ 0x80) as i8);
    let s_90 = _mm512_set1_epi8((0x90u8 ^ 0x80) as i8);
    let s_88 = _mm512_set1_epi8((0x88u8 ^ 0x80) as i8);
    let s_84 = _mm512_set1_epi8((0x84u8 ^ 0x80) as i8);

    let mut i = 0usize;
    // Five bytes of lookahead are needed for a six-byte sequence.
    while i + LANE + 5 <= data.len() {
        let chunk = _mm512_loadu_si512(data.as_ptr().add(i) as *const __m512i);
        let hi = _mm512_movepi8_mask(chunk);

        if hi == 0 {
            chars += LANE as u64;
            i += LANE;
            continue;
        }

        let n1 = _mm512_loadu_si512(data.as_ptr().add(i + 1) as *const __m512i);
        let n2 = _mm512_loadu_si512(data.as_ptr().add(i + 2) as *const __m512i);

        // A continuation byte is 0x80..=0xBF, i.e. biased value <= 0x3F.
        let is_cont = |v: __m512i| _mm512_cmple_epu8_mask(_mm512_xor_si512(v, bias), v_3f);
        let c1 = is_cont(n1);
        let c2 = is_cont(n2);

        // Sequence length implied by the lead byte.
        let b = _mm512_xor_si512(chunk, bias);
        let ge_e0 = _mm512_cmpgt_epi8_mask(b, t_e0);
        let ge_f0 = _mm512_cmpgt_epi8_mask(b, t_f0);

        let ascii = !hi;
        // 0xC2..=0xFD are the only usable leads.
        let lead = _mm512_cmpgt_epu8_mask(chunk, v_c1) & _mm512_cmple_epu8_mask(chunk, v_fd);

        let l2 = lead & !ge_e0;
        let l3 = ge_e0 & !ge_f0;

        // Two- and three-byte sequences cover everything a real encoder emits.
        let mut ok_len = (l2 & c1) | (l3 & c1 & c2);

        // Four-, five- and six-byte forms need more lookahead than the common
        // case, so the extra loads happen only on lanes that contain such a
        // lead byte.
        if ge_f0 != 0 {
            let ge_f8 = _mm512_cmpgt_epi8_mask(b, t_f8);
            let ge_fc = _mm512_cmpgt_epi8_mask(b, t_fc);
            let c3 = is_cont(_mm512_loadu_si512(data.as_ptr().add(i + 3) as *const __m512i));
            let c4 = is_cont(_mm512_loadu_si512(data.as_ptr().add(i + 4) as *const __m512i));
            let c5 = is_cont(_mm512_loadu_si512(data.as_ptr().add(i + 5) as *const __m512i));
            let l4 = ge_f0 & !ge_f8;
            let l5 = ge_f8 & !ge_fc;
            let l6 = ge_fc & lead;
            ok_len |= (l4 & c1 & c2 & c3)
                | (l5 & c1 & c2 & c3 & c4)
                | (l6 & c1 & c2 & c3 & c4 & c5);
        }

        // Overlong forms and surrogates are settled by the second byte after
        // one of five specific lead bytes.
        let f = _mm512_xor_si512(n1, bias);
        let bad = (_mm512_cmpeq_epi8_mask(chunk, e0) & _mm512_cmplt_epi8_mask(f, s_a0))
            | (_mm512_cmpeq_epi8_mask(chunk, ed) & _mm512_cmpge_epi8_mask(f, s_a0))
            | (_mm512_cmpeq_epi8_mask(chunk, f0) & _mm512_cmplt_epi8_mask(f, s_90))
            | (_mm512_cmpeq_epi8_mask(chunk, f8) & _mm512_cmplt_epi8_mask(f, s_88))
            | (_mm512_cmpeq_epi8_mask(chunk, fc) & _mm512_cmplt_epi8_mask(f, s_84));
        let starts = (ascii | ok_len) & !bad;
        chars += starts.count_ones() as u64;
        i += LANE;
    }

    chars + chars_tail_scalar(data, i)
}

/// Per-position character count for the bytes a vector lane cannot cover.
#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_chars_only_avx2(data: &[u8], mode: WsMode) -> u64 {
    const LANE: usize = 32;
    let _ = mode;

    // Same per-position rule as the AVX-512 kernel; see the comment there for
    // why no state crosses a lane and no scalar fallback is needed.
    let mut chars = 0u64;

    let bias = _mm256_set1_epi8(0x80u8 as i8);
    let v_c0 = _mm256_set1_epi8(0xc0u8 as i8);
    let v_c1b = _mm256_set1_epi8((0xC1u8 ^ 0x80) as i8);
    let v_fdb = _mm256_set1_epi8((0xFDu8 ^ 0x80) as i8);

    let t_e0 = _mm256_set1_epi8((0xE0u8 ^ 0x80) as i8 - 1);
    let t_f0 = _mm256_set1_epi8((0xF0u8 ^ 0x80) as i8 - 1);
    let t_f8 = _mm256_set1_epi8((0xF8u8 ^ 0x80) as i8 - 1);
    let t_fc = _mm256_set1_epi8((0xFCu8 ^ 0x80) as i8 - 1);

    let e0 = _mm256_set1_epi8(0xe0u8 as i8);
    let ed = _mm256_set1_epi8(0xedu8 as i8);
    let f0 = _mm256_set1_epi8(0xf0u8 as i8);
    let f8 = _mm256_set1_epi8(0xf8u8 as i8);
    let fc = _mm256_set1_epi8(0xfcu8 as i8);

    let s_a0 = _mm256_set1_epi8((0xA0u8 ^ 0x80) as i8);
    let s_90 = _mm256_set1_epi8((0x90u8 ^ 0x80) as i8);
    let s_88 = _mm256_set1_epi8((0x88u8 ^ 0x80) as i8);
    let s_84 = _mm256_set1_epi8((0x84u8 ^ 0x80) as i8);

    let mut i = 0usize;
    while i + LANE + 5 <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        let hi = _mm256_movemask_epi8(chunk) as u32;

        if hi == 0 {
            chars += LANE as u64;
            i += LANE;
            continue;
        }

        // A continuation byte is exactly 0x80..=0xBF, i.e. (b & 0xC0) == 0x80.
        // Masking and comparing for equality avoids the signed-compare trap
        // that a range test on 256-bit lanes would otherwise fall into.
        let cont_of = |off: usize| -> u32 {
            let v = _mm256_loadu_si256(data.as_ptr().add(i + off) as *const __m256i);
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(v, v_c0), bias)) as u32
        };
        let c1 = cont_of(1);
        let c2 = cont_of(2);
        let c3 = cont_of(3);
        let c4 = cont_of(4);
        let c5 = cont_of(5);

        let b = _mm256_xor_si256(chunk, bias);
        let mm = |v: __m256i| _mm256_movemask_epi8(v) as u32;
        let ge_e0 = mm(_mm256_cmpgt_epi8(b, t_e0));
        let ge_f0 = mm(_mm256_cmpgt_epi8(b, t_f0));
        let ge_f8 = mm(_mm256_cmpgt_epi8(b, t_f8));
        let ge_fc = mm(_mm256_cmpgt_epi8(b, t_fc));

        let ascii = !hi;
        let lead = mm(_mm256_cmpgt_epi8(b, v_c1b)) & !mm(_mm256_cmpgt_epi8(b, v_fdb));

        let l2 = lead & !ge_e0;
        let l3 = ge_e0 & !ge_f0;
        let l4 = ge_f0 & !ge_f8;
        let l5 = ge_f8 & !ge_fc;
        let l6 = ge_fc & lead;

        let ok_len = (l2 & c1)
            | (l3 & c1 & c2)
            | (l4 & c1 & c2 & c3)
            | (l5 & c1 & c2 & c3 & c4)
            | (l6 & c1 & c2 & c3 & c4 & c5);

        let n1 = _mm256_loadu_si256(data.as_ptr().add(i + 1) as *const __m256i);
        let f = _mm256_xor_si256(n1, bias);
        let bad_special = (mm(_mm256_cmpeq_epi8(chunk, e0)) & mm(_mm256_cmpgt_epi8(s_a0, f)))
            | (mm(_mm256_cmpeq_epi8(chunk, ed)) & !mm(_mm256_cmpgt_epi8(s_a0, f)))
            | (mm(_mm256_cmpeq_epi8(chunk, f0)) & mm(_mm256_cmpgt_epi8(s_90, f)))
            | (mm(_mm256_cmpeq_epi8(chunk, f8)) & mm(_mm256_cmpgt_epi8(s_88, f)))
            | (mm(_mm256_cmpeq_epi8(chunk, fc)) & mm(_mm256_cmpgt_epi8(s_84, f)));

        let starts = ascii | (ok_len & !bad_special);
        chars += starts.count_ones() as u64;
        i += LANE;
    }

    chars + chars_tail_scalar(data, i)
}

/// Positions of multi-byte whitespace characters, as AVX-512 masks.
///
/// Same classification as the AVX2 form, but every compare lands straight in a
/// mask register, so the combining is plain 64-bit integer arithmetic.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn ws_seq_masks512<const NBSP: bool>(
    chunk: __m512i,
    next1: __m512i,
    next2: __m512i,
) -> (u64, u64) {
    let high = _mm512_set1_epi8(0x80u8 as i8);

    let is_c2 = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(ws::LEAD_C2 as i8));
    let is_e1 = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(ws::LEAD_E1 as i8));
    let is_e2 = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(ws::LEAD_E2 as i8));
    let is_e3 = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(ws::LEAD_E3 as i8));

    let at1_80 = _mm512_cmpeq_epi8_mask(next1, _mm512_set1_epi8(0x80u8 as i8));
    let at1_81 = _mm512_cmpeq_epi8_mask(next1, _mm512_set1_epi8(0x81u8 as i8));
    let at1_9a = _mm512_cmpeq_epi8_mask(next1, _mm512_set1_epi8(0x9au8 as i8));
    let at2_80 = _mm512_cmpeq_epi8_mask(next2, _mm512_set1_epi8(0x80u8 as i8));

    let ws2 = if NBSP {
        is_c2 & _mm512_cmpeq_epi8_mask(next1, _mm512_set1_epi8(0xa0u8 as i8))
    } else {
        0
    };

    let tail = _mm512_xor_si512(next2, high);

    let mut low = _mm512_cmpgt_epi8_mask(_mm512_set1_epi8(0x0b), tail)
        & _mm512_cmpgt_epi8_mask(tail, _mm512_set1_epi8(-1));
    if !NBSP {
        low &= !_mm512_cmpeq_epi8_mask(tail, _mm512_set1_epi8(0x07));
    }

    let mut sep = _mm512_cmpeq_epi8_mask(
        _mm512_and_si512(tail, _mm512_set1_epi8(0xfeu8 as i8)),
        _mm512_set1_epi8(0x28),
    );
    if NBSP {
        sep |= _mm512_cmpeq_epi8_mask(tail, _mm512_set1_epi8(0x2f));
    }

    let mut narrow = _mm512_cmpeq_epi8_mask(tail, _mm512_set1_epi8(0x1f));
    if NBSP {
        narrow |= _mm512_cmpeq_epi8_mask(tail, _mm512_set1_epi8(0x20));
    }

    let e2_hit = (at1_80 & (low | sep)) | (at1_81 & narrow);
    let ws3 = (is_e2 & e2_hit) | (at2_80 & ((is_e1 & at1_9a) | (is_e3 & at1_80)));

    (ws2, ws3)
}

/// AVX-512 kernel for lines and words.
///
/// Lines and words never need decoding or validation, so the 64-byte lane
/// carries no extra cost over the 32-byte one while halving the number of
/// iterations and the branch-prediction pressure that goes with them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn count_lw_avx512(
    data: &[u8],
    carry_in: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    const LANE: usize = 64;

    let mut lines = 0u64;
    let mut words = 0u64;
    let mut carry = carry_in;
    let mut ws_carry = 0u64;

    let newline = _mm512_set1_epi8(b'\n' as i8);

    // Byte classification is done with nibble lookups instead of a chain of
    // broadcast compares. Each table is indexed by one nibble of the byte and
    // yields a bitmask; ANDing the high- and low-nibble results identifies the
    // byte. This keeps the work in the vector domain, where several ports can
    // issue it, rather than in the mask domain, which is a single port and was
    // the limit for this loop.
    //
    // ascii table: bit0 = the \t \n \v \f \r group, bit1 = space.
    let ws_hi_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0x01, 0, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ));
    let ws_lo_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x01, 0x01, 0x01, 0x01, 0, 0,
    ));

    // lead table: C2 = bit0, E1 = bit1, E2 = bit2, E3 = bit3. The C2 entry is
    // dropped when the locale does not treat NBSP as a separator.
    let lead_hi_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, if mode.nbsp { 0x01 } else { 0 }, 0, 0x0e, 0,
    ));
    let lead_lo_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0x02, 0x05, 0x08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ));

    // second-byte table, using the same bit per lead, so a lead and the byte
    // after it share a bit only when they can form a whitespace character.
    let sec_hi_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0x0c, 0x02, 0x01, 0, 0, 0, 0, 0,
    ));
    let sec_lo_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0x0d, 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0, 0, 0, 0, 0,
    ));

    let low_nibble = _mm512_set1_epi8(0x0f);
    let zero = _mm512_setzero_si512();

    let mut i = 0usize;
    while i + LANE + 3 <= data.len() {
        let chunk = _mm512_loadu_si512(data.as_ptr().add(i) as *const __m512i);

        let nl_bits = _mm512_cmpeq_epi8_mask(chunk, newline);

        let hi_idx = _mm512_and_si512(_mm512_srli_epi16(chunk, 4), low_nibble);
        let lo_idx = _mm512_and_si512(chunk, low_nibble);

        // A byte with the high bit set must not alias a table entry, so the
        // high-nibble lookups are masked back to zero for those lanes.
        let ascii_hit = _mm512_and_si512(
            _mm512_shuffle_epi8(ws_hi_lut, hi_idx),
            _mm512_shuffle_epi8(ws_lo_lut, lo_idx),
        );
        let ascii_ws_bits = _mm512_cmpneq_epi8_mask(ascii_hit, zero);

        // The second byte of a character at j sits at j + 1, so the lane is
        // reloaded at a one-byte offset and the two classifications are ANDed.
        let next1 = _mm512_loadu_si512(data.as_ptr().add(i + 1) as *const __m512i);
        let nhi_idx = _mm512_and_si512(_mm512_srli_epi16(next1, 4), low_nibble);
        let nlo_idx = _mm512_and_si512(next1, low_nibble);

        let lead_hit = _mm512_and_si512(
            _mm512_shuffle_epi8(lead_hi_lut, hi_idx),
            _mm512_shuffle_epi8(lead_lo_lut, lo_idx),
        );
        let sec_hit = _mm512_and_si512(
            _mm512_shuffle_epi8(sec_hi_lut, nhi_idx),
            _mm512_shuffle_epi8(sec_lo_lut, nlo_idx),
        );
        let cand = _mm512_test_epi8_mask(lead_hit, sec_hit);

        let ws_all;
        if cand == 0 {
            ws_all = ascii_ws_bits | ws_carry;
            ws_carry = 0;
        } else {
            let next2 = _mm512_loadu_si512(data.as_ptr().add(i + 2) as *const __m512i);
            let (ws2, ws3) = if mode.nbsp {
                ws_seq_masks512::<true>(chunk, next1, next2)
            } else {
                ws_seq_masks512::<false>(chunk, next1, next2)
            };
            let wide = ((ws2 as u128) | ((ws2 as u128) << 1))
                | ((ws3 as u128) | ((ws3 as u128) << 1) | ((ws3 as u128) << 2));
            ws_all = ascii_ws_bits | (wide as u64) | ws_carry;
            ws_carry = (wide >> LANE) as u64;
        }

        lines += nl_bits.count_ones() as u64;
        let prev_ws_bits = (ws_all << 1) | (carry as u64);
        words += (!ws_all & prev_ws_bits).count_ones() as u64;
        carry = (ws_all >> (LANE - 1)) & 1 == 1;
        i += LANE;
    }

    i += ws_carry.count_ones() as usize;

    let (t_lines, t_words, _t_bytes, _t_chars, t_carry) =
        ws::count_scalar_unicode(&data[i..], carry, false, mode);

    (lines + t_lines, words + t_words, data.len() as u64, 0, t_carry)
}

/// Mask of positions holding a lead byte of a non-ASCII whitespace character.
///
/// Used only as a cheap gate: a lane with no candidate lead cannot contain
/// multi-byte white space, so the full classification can be skipped.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ws_lead_candidates<const NBSP: bool>(chunk: __m256i) -> i32 {
    let cand = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E1 as i8)),
            _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E2 as i8)),
        ),
        if NBSP {
            _mm256_or_si256(
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E3 as i8)),
                _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_C2 as i8)),
            )
        } else {
            _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(ws::LEAD_E3 as i8))
        },
    );
    _mm256_movemask_epi8(cand)
}

/// AVX2 kernel that is correct for Unicode whitespace.
///
/// A lane with no high bit takes the ASCII path unchanged. A lane that has one
/// stays in SIMD: bytes are classified into leads and continuations, the lane
/// is validated as UTF-8, and the multi-byte whitespace characters are matched
/// with compares against neighbouring offsets. Only malformed input reaches the
/// scalar decoder.
#[target_feature(enable = "avx2")]
unsafe fn count_buf_avx2_unicode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    const LANE: usize = 32;

    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let mut carry = carry_in;
    let mut ws_carry = 0u32;
    let mut cont_pending = 0u64;

    let newline = _mm256_set1_epi8(b'\n' as i8);
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let vtab = _mm256_set1_epi8(0x0b);
    let ff = _mm256_set1_epi8(0x0c);
    let cr = _mm256_set1_epi8(b'\r' as i8);

    let high = _mm256_set1_epi8(0x80u8 as i8);
    let cont_mask = _mm256_set1_epi8(0xC0u8 as i8);

    // Lead-byte classification runs on `byte ^ 0x80`, which turns the unsigned
    // byte order into the signed order that `cmpgt` uses.
    let v41 = _mm256_set1_epi8(0x41);
    let v5f = _mm256_set1_epi8(0x5f);
    let v6f = _mm256_set1_epi8(0x6f);
    let v77 = _mm256_set1_epi8(0x77);

    let lead_e0 = _mm256_set1_epi8(0xE0u8 as i8);
    let lead_ed = _mm256_set1_epi8(0xEDu8 as i8);
    let lead_f0 = _mm256_set1_epi8(0xF0u8 as i8);
    let lead_f4 = _mm256_set1_epi8(0xF4u8 as i8);

    let v0f = _mm256_set1_epi8(0x0f);
    let v10 = _mm256_set1_epi8(0x10);
    let v1f = _mm256_set1_epi8(0x1f);
    let v20 = _mm256_set1_epi8(0x20);

    let mut i = 0usize;
    while i + LANE + 3 <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

        let eq_nl = _mm256_cmpeq_epi8(chunk, newline);
        let ws_vec = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(eq_nl, _mm256_cmpeq_epi8(chunk, space)),
                _mm256_or_si256(_mm256_cmpeq_epi8(chunk, tab), _mm256_cmpeq_epi8(chunk, vtab)),
            ),
            _mm256_or_si256(_mm256_cmpeq_epi8(chunk, ff), _mm256_cmpeq_epi8(chunk, cr)),
        );

        let nl_bits = _mm256_movemask_epi8(eq_nl) as u32;
        let ascii_ws_bits = _mm256_movemask_epi8(ws_vec) as u32;
        let hi_bits = _mm256_movemask_epi8(chunk) as u32;

        if want_chars && hi_bits == 0 {
            let ascii_ws_bits = ascii_ws_bits | ws_carry;
            ws_carry = 0;
            lines += nl_bits.count_ones() as u64;
            let prev_ws_bits = (ascii_ws_bits << 1) | (carry as u32);
            words += (!ascii_ws_bits & prev_ws_bits).count_ones() as u64;
            carry = (ascii_ws_bits >> (LANE - 1)) & 1 == 1;
            chars += LANE as u64;
            i += LANE;
            continue;
        }

        if !want_chars {
            // Lines and words never need decoding or validation. Every byte of
            // a non-whitespace character is itself non-whitespace, so counting
            // transitions per byte matches counting them per character, and the
            // whitespace lead bytes cannot occur inside any other sequence.
            //
            // An all-ASCII lane is not special-cased ahead of this test: it is
            // subsumed by `cand == 0`, and on text that mixes scripts the
            // extra branch mispredicts on nearly every lane while saving only
            // four compares.
            let cand = if mode.nbsp {
                ws_lead_candidates::<true>(chunk)
            } else {
                ws_lead_candidates::<false>(chunk)
            };
            if cand == 0 {
                let ws_all = ascii_ws_bits | ws_carry;
                ws_carry = 0;
                lines += nl_bits.count_ones() as u64;
                let prev_ws_bits = (ws_all << 1) | (carry as u32);
                words += (!ws_all & prev_ws_bits).count_ones() as u64;
                carry = (ws_all >> (LANE - 1)) & 1 == 1;
                i += LANE;
                continue;
            }

            let next1 = _mm256_loadu_si256(data.as_ptr().add(i + 1) as *const __m256i);
            let next2 = _mm256_loadu_si256(data.as_ptr().add(i + 2) as *const __m256i);
            let (ws2, ws3) = if mode.nbsp {
                ws_seq_masks::<true>(chunk, next1, next2)
            } else {
                ws_seq_masks::<false>(chunk, next1, next2)
            };

            // A whitespace character can start near the lane end and finish in
            // the next one. Rather than shortening the stride, the overhanging
            // bits are carried forward, which keeps the loop at a constant 32
            // bytes per iteration.
            let wide = ((ws2 as u64) | ((ws2 as u64) << 1))
                | ((ws3 as u64) | ((ws3 as u64) << 1) | ((ws3 as u64) << 2));
            let ws_all = ascii_ws_bits | (wide as u32) | ws_carry;
            ws_carry = (wide >> LANE) as u32;

            lines += nl_bits.count_ones() as u64;
            let prev_ws_bits = (ws_all << 1) | (carry as u32);
            words += (!ws_all & prev_ws_bits).count_ones() as u64;
            carry = (ws_all >> (LANE - 1)) & 1 == 1;
            i += LANE;
            continue;
        }

        let next1 = _mm256_loadu_si256(data.as_ptr().add(i + 1) as *const __m256i);
        let next2 = _mm256_loadu_si256(data.as_ptr().add(i + 2) as *const __m256i);

        // Classify sequence lengths as bitmasks rather than vectors: three
        // threshold compares give every boundary, and the rest is integer
        // work. Comparisons are signed, so bias the bytes by 0x80 first.
        let shifted = _mm256_xor_si256(chunk, high);
        let m_ge_c2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(shifted, v41)) as u32;
        let m_ge_e0 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(shifted, v5f)) as u32;
        let m_ge_f0 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(shifted, v6f)) as u32;
        let m_ge_f8 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(shifted, v77)) as u32;
        let m_cont =
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(chunk, cont_mask), high)) as u32;

        let m_lead = hi_bits & !m_cont;
        let m_l2 = m_lead & !m_ge_e0;
        let m_l3 = m_ge_e0 & !m_ge_f0;
        let m_l4 = m_ge_f0 & !m_ge_f8;
        let m_bad = m_ge_f8 | (m_lead & !m_ge_c2);

        // Every lead is checked eagerly against the bytes that follow it, so a
        // sequence crossing the lane end is settled here instead of by
        // shortening the stride. Deferring it would be wrong: the next lane may
        // take the all-ASCII path, which never revisits pending requirements.
        // The shifted masks come from m_cont plus the three bytes past the
        // lane, which is cheaper than loading and testing those bytes again.
        let o0 = ((data[i + LANE] & 0xC0) == 0x80) as u32;
        let o1 = ((data[i + LANE + 1] & 0xC0) == 0x80) as u32;
        let o2 = ((data[i + LANE + 2] & 0xC0) == 0x80) as u32;
        let c1 = (m_cont >> 1) | (o0 << (LANE - 1));
        let c2 = (m_cont >> 2) | (o0 << (LANE - 2)) | (o1 << (LANE - 1));
        let c3 = (m_cont >> 3) | (o0 << (LANE - 3)) | (o1 << (LANE - 2)) | (o2 << (LANE - 1));
        let bad_lead = (m_l2 & !c1) | (m_l3 & !(c1 & c2)) | (m_l4 & !(c1 & c2 & c3));

        // Continuation bytes are legal only where a lead put them, counting the
        // leads left over from the previous lane.
        let l34 = (m_l3 | m_l4) as u64;
        let cover = (((m_l2 as u64) | l34) << 1) | (l34 << 2) | ((m_l4 as u64) << 3);
        let expected = cover | cont_pending;
        let orphan = (m_cont as u64) & !expected;

        let mut valid = m_bad == 0 && bad_lead == 0 && orphan == 0;

        // Overlong, surrogate and out-of-range forms all begin with a 3- or
        // 4-byte lead, so two-byte-only lanes can skip the check entirely.
        if valid && (m_l3 | m_l4) != 0 {
            let special = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(chunk, lead_e0),
                    _mm256_cmpeq_epi8(chunk, lead_ed),
                ),
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(chunk, lead_f0),
                    _mm256_cmpeq_epi8(chunk, lead_f4),
                ),
            );
            if _mm256_movemask_epi8(special) != 0 {
                // Overlong, surrogate and out-of-range sequences are the only
                // remaining way a structurally sound lane can be malformed.
                let follow = _mm256_xor_si256(next1, high);
                let rejected = _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_and_si256(
                            _mm256_cmpeq_epi8(chunk, lead_e0),
                            _mm256_cmpgt_epi8(v20, follow),
                        ),
                        _mm256_and_si256(
                            _mm256_cmpeq_epi8(chunk, lead_ed),
                            _mm256_cmpgt_epi8(follow, v1f),
                        ),
                    ),
                    _mm256_or_si256(
                        _mm256_and_si256(
                            _mm256_cmpeq_epi8(chunk, lead_f0),
                            _mm256_cmpgt_epi8(v10, follow),
                        ),
                        _mm256_and_si256(
                            _mm256_cmpeq_epi8(chunk, lead_f4),
                            _mm256_cmpgt_epi8(follow, v0f),
                        ),
                    ),
                );
                if _mm256_movemask_epi8(rejected) != 0 {
                    valid = false;
                }
            }
        }

        if !valid {
            // Resume after any bytes belonging to a character the previous
            // lane already accounted for, so the decoder starts on a boundary.
            i += cont_pending.count_ones() as usize;
            cont_pending = 0;
            ws_carry = 0;
            let mut end = (i + LANE).min(data.len());
            while end < data.len() && (data[end] & 0xC0) == 0x80 {
                end += 1;
            }
            let (l, w, _b, c, carry_out) =
                ws::count_scalar_unicode(&data[i..end], carry, want_chars, mode);
            lines += l;
            words += w;
            chars += c;
            carry = carry_out;
            i = end;
            continue;
        }

        let cand = if mode.nbsp {
            ws_lead_candidates::<true>(chunk)
        } else {
            ws_lead_candidates::<false>(chunk)
        };
        let (ws2, ws3) = if cand == 0 {
            (0, 0)
        } else if mode.nbsp {
            ws_seq_masks::<true>(chunk, next1, next2)
        } else {
            ws_seq_masks::<false>(chunk, next1, next2)
        };

        let wide = ((ws2 as u64) | ((ws2 as u64) << 1))
            | ((ws3 as u64) | ((ws3 as u64) << 1) | ((ws3 as u64) << 2));
        let ws_all = ascii_ws_bits | (wide as u32) | ws_carry;
        ws_carry = (wide >> LANE) as u32;
        cont_pending = expected >> LANE;

        let starts = !m_cont;

        lines += nl_bits.count_ones() as u64;
        let prev_ws_bits = (ws_all << 1) | (carry as u32);
        words += (starts & !ws_all & prev_ws_bits).count_ones() as u64;
        chars += starts.count_ones() as u64;
        carry = (ws_all >> (LANE - 1)) & 1 == 1;
        i += LANE;
    }

    // A character may span the end of the last full lane; its remaining bytes
    // were already accounted for, so skip them and let the tail start on a
    // character boundary.
    i += if want_chars { cont_pending.count_ones() } else { ws_carry.count_ones() } as usize;

    let (t_lines, t_words, _t_bytes, t_chars, t_carry) =
        ws::count_scalar_unicode(&data[i..], carry, want_chars, mode);

    lines += t_lines;
    words += t_words;
    chars += t_chars;

    (lines, words, data.len() as u64, chars, t_carry)
}

/// Measure the run of bytes at `data[start..]` that lie in 0x20..=0x7E.
///
/// Every byte in that range is a self-contained character of display width
/// one that never ends a line, never needs decoding and never affects the
/// running maximum, so an entire run collapses into two additions on
/// `linepos` and `chars`. Only the word transitions still have to be
/// counted, and those come straight off the space bitmask.
///
/// `carry_ws` is true when the preceding character was white space. Returns
/// the end offset, the words that began inside the run, and whether the run
/// ended on a space.
pub fn simple_ascii_run(data: &[u8], start: usize, carry_ws: bool, avx2: bool) -> (usize, u64, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2 {
            return unsafe { simple_ascii_run_avx2(data, start, carry_ws) };
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = avx2;
    simple_ascii_run_scalar(data, start, carry_ws)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simple_ascii_run_avx2(data: &[u8], start: usize, carry_ws: bool) -> (usize, u64, bool) {
    const LANE: usize = 32;

    let mut i = start;
    let mut words = 0u64;
    let mut carry = carry_ws;

    let space = _mm256_set1_epi8(b' ' as i8);
    let v20 = _mm256_set1_epi8(0x20);
    let v7e = _mm256_set1_epi8(0x7e);

    while i + LANE <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

        // Unsigned range test without shifting the sign bit around: a byte is
        // in range exactly when clamping it to the bounds leaves it alone.
        let ge20 = _mm256_cmpeq_epi8(_mm256_max_epu8(chunk, v20), chunk);
        let le7e = _mm256_cmpeq_epi8(_mm256_min_epu8(chunk, v7e), chunk);
        let simple = _mm256_and_si256(ge20, le7e);

        let bad = !(_mm256_movemask_epi8(simple) as u32);
        let sp = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, space)) as u32;

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
    }

    let (end, w, carry) = simple_ascii_run_scalar(data, i, carry);
    (end, words + w, carry)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
        count_buf_scalar(data, carry_in, want_chars)
    }

    #[test]
    fn avx2_unicode_agrees_on_every_codepoint() {
        if !avx2_available() {
            return;
        }
        // Every scalar value, at every offset within a lane, so the vector
        // whitespace matching is checked against the decoder exhaustively.
        for modes in [
            WsMode { unicode: true, nbsp: true },
            WsMode { unicode: true, nbsp: false },
        ] {
            let mut buf = [0u8; 96];
            for cp in 0u32..=0x10FFFF {
                let ch = match char::from_u32(cp) {
                    Some(c) => c,
                    None => continue,
                };
                let mut enc = [0u8; 4];
                let seq = ch.encode_utf8(&mut enc).as_bytes();
                for off in [0usize, 1, 29, 30, 31, 32, 33, 62, 63] {
                    buf.fill(b'x');
                    buf[off..off + seq.len()].copy_from_slice(seq);
                    let expected = ws::count_scalar_unicode(&buf, false, true, modes);
                    let actual = unsafe { count_buf_avx2_unicode(&buf, false, true, modes) };
                    assert_eq!(expected, actual, "cp=U+{cp:04X} off={off} mode={modes:?}");
                }
            }
        }
    }

    #[test]
    fn avx2_unicode_matches_scalar_on_arbitrary_bytes() {
        if !avx2_available() {
            return;
        }
        // Random bytes exercise malformed, truncated and overlong sequences at
        // every alignment, which the curated-piece generator cannot reach.
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for modes in [
            WsMode { unicode: true, nbsp: true },
            WsMode { unicode: true, nbsp: false },
        ] {
            for len in 0..140usize {
                for _ in 0..40 {
                    let mut data = vec![0u8; len];
                    for b in data.iter_mut() {
                        let r = next();
                        // Bias towards lead bytes and continuations so that
                        // sequence structure, not just noise, is generated.
                        *b = match r % 4 {
                            0 => (r >> 8) as u8,
                            1 => 0x80 | ((r >> 8) as u8 & 0x3f),
                            2 => [0xc2, 0xe1, 0xe2, 0xe3, 0xe0, 0xed, 0xf0, 0xf4][(r >> 8) as usize % 8],
                            _ => [b'a', b' ', b'\n', b'\t'][(r >> 8) as usize % 4],
                        };
                    }
                    for carry_in in [true, false] {
                        for want_chars in [true, false] {
                            let expected =
                                ws::count_scalar_unicode(&data, carry_in, want_chars, modes);
                            let actual = unsafe {
                                count_buf_avx2_unicode(&data, carry_in, want_chars, modes)
                            };
                            assert_eq!(
                                expected, actual,
                                "mismatch data={data:02x?} carry_in={carry_in} want_chars={want_chars} mode={modes:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn avx2_unicode_matches_scalar_on_random_inputs() {
        if !avx2_available() {
            return;
        }
        let pieces: &[&[u8]] = &[
            b"a", b"b", b" ", b"\t", b"\n",
            "\u{00A0}".as_bytes(), "\u{2003}".as_bytes(), "\u{3000}".as_bytes(),
            "\u{1680}".as_bytes(), "\u{2007}".as_bytes(), "\u{202F}".as_bytes(),
            "\u{2060}".as_bytes(), "\u{200B}".as_bytes(), "\u{00E9}".as_bytes(),
            "\u{4E2D}".as_bytes(), b"\xff", b"\xc2",
        ];
        let mut seed: u64 = 0xdead_beef_1234_5678;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for modes in [
            WsMode { unicode: true, nbsp: true },
            WsMode { unicode: true, nbsp: false },
        ] {
            for _ in 0..200 {
                let n = (next() as usize) % 300;
                let mut data = Vec::new();
                for _ in 0..n {
                    data.extend_from_slice(pieces[(next() as usize) % pieces.len()]);
                }
                for carry_in in [true, false] {
                    for want_chars in [true, false] {
                        let expected =
                            ws::count_scalar_unicode(&data, carry_in, want_chars, modes);
                        let actual = unsafe {
                            count_buf_avx2_unicode(&data, carry_in, want_chars, modes)
                        };
                        assert_eq!(
                            expected, actual,
                            "mismatch len={} carry_in={carry_in} want_chars={want_chars} mode={modes:?}",
                            data.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unicode_kernel_matches_ascii_kernel_on_ascii() {
        if !avx2_available() {
            return;
        }
        let data: Vec<u8> = b"the quick brown fox\tjumps\nover the lazy dog  again and again "
            .iter()
            .cycle()
            .take(5000)
            .copied()
            .collect();
        let mode = WsMode { unicode: true, nbsp: true };
        for carry_in in [true, false] {
            let a = unsafe { count_buf_avx2(&data, carry_in, true) };
            let b = unsafe { count_buf_avx2_unicode(&data, carry_in, true, mode) };
            assert_eq!(a, b);
        }
    }

    #[test]
    fn avx2_matches_scalar_on_random_inputs() {
        if !avx2_available() {
            return;
        }
        let alphabet: &[u8] = b"abc \t\n\r\x0b\x0cXYZ\xC3\xA9\xE2\x98\x83";
        let mut seed: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for len in [0usize, 1, 31, 32, 33, 63, 64, 65, 257, 4099] {
            let data: Vec<u8> = (0..len)
                .map(|_| alphabet[(next() as usize) % alphabet.len()])
                .collect();
            for carry_in in [true, false] {
                for want_chars in [true, false] {
                    let expected = reference(&data, carry_in, want_chars);
                    let actual = unsafe { count_buf_avx2(&data, carry_in, want_chars) };
                    assert_eq!(
                        expected, actual,
                        "mismatch at len={len} carry_in={carry_in} want_chars={want_chars}"
                    );
                }
            }
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
    fn avx512_chars_match_avx2_and_scalar() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        let mut st = 0x12345678u64;
        let mut rnd = move || {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17; st
        };
        let pieces: [&[u8]; 14] = [
            b"a", b" ", b"\n", b"\xc3\xa9", b"\xe4\xb8\xad", b"\xd0\x96",
            b"\xf0\x9f\x98\x80", b"\xff", b"\xc0\xa0", b"\xed\xa0\x80",
            b"\xf8\x87\xbf\xbf\xbf", b"\x80", b"\xe0\x80\x80", b"\xf4\x90\x80\x80",
        ];
        for mode in [WsMode { unicode: true, nbsp: true }, WsMode { unicode: true, nbsp: false }] {
            for _ in 0..4000 {
                let n = (rnd() % 400) as usize;
                let mut data = Vec::new();
                while data.len() < n {
                    data.extend_from_slice(pieces[(rnd() % 14) as usize]);
                }
                let scalar = ws::count_scalar_unicode(&data, true, true, mode).3;
                let a2 = unsafe { count_chars_only_avx2(&data, mode) };
                let a5 = unsafe { count_chars_only_avx512(&data, mode) };
                assert_eq!(scalar, a2, "avx2 mismatch len={} {:02x?}", data.len(), &data);
                assert_eq!(scalar, a5, "avx512 mismatch len={} {:02x?}", data.len(), &data);
            }
        }
    }

    #[test]
    fn avx512_lines_words_match_scalar() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        let mut st = 0x9e3779b9u64;
        let mut rnd = move || {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17; st
        };
        let pieces: [&[u8]; 20] = [
            b"a", b"word", b" ", b"\n", b"\t", b"\r", b"\x0b", b"\x0c",
            b"\xc2\xa0", b"\xe2\x80\x80", b"\xe2\x80\x87", b"\xe2\x80\xaf",
            b"\xe2\x81\xa0", b"\xe2\x81\x9f", b"\xe3\x80\x80", b"\xe1\x9a\x80",
            b"\xe2\x80\xa8", b"\xe4\xb8\xad", b"\xff", b"\xc2",
        ];
        for mode in [WsMode { unicode: true, nbsp: true }, WsMode { unicode: true, nbsp: false }] {
            for carry_in in [true, false] {
                for _ in 0..3000 {
                    let n = (rnd() % 400) as usize;
                    let mut data = Vec::new();
                    while data.len() < n {
                        data.extend_from_slice(pieces[(rnd() % 20) as usize]);
                    }
                    let (sl, sw, _sb, _sc, sk) =
                        ws::count_scalar_unicode(&data, carry_in, false, mode);
                    let (al, aw, _ab, _ac, ak) =
                        unsafe { count_lw_avx512(&data, carry_in, mode) };
                    assert_eq!(
                        (sl, sw, sk), (al, aw, ak),
                        "avx512 lw mismatch carry={carry_in} mode={mode:?} data={:02x?}", &data
                    );
                }
            }
        }
    }

    /// Character counting judges every byte position on its own merits, so a
    /// malformed sequence must cost exactly what a clean one costs and must
    /// never disturb the positions around it. Arbitrary bytes are the only way
    /// to reach the truncated, overlong and surrogate forms that text never
    /// contains, and the pieces below are deliberately placed so that lead
    /// bytes land on and straddle every lane boundary.
    #[test]
    fn chars_only_kernels_match_scalar_on_arbitrary_bytes() {
        let pieces: [&[u8]; 24] = [
            b"a",
            b"abcdefgh",
            "\u{00e9}".as_bytes(),
            "\u{4e00}".as_bytes(),
            "\u{1f600}".as_bytes(),
            b"\xc2",
            b"\xc1\xa0",
            b"\xe0\x80\x80",
            b"\xed\xa0\x80",
            b"\xf0\x80\x80\x80",
            b"\xf8\x80\x80\x80\x80",
            b"\xfc\x84\x80\x80\x80\x80",
            b"\xfd\xbf\xbf\xbf\xbf\xbf",
            b"\xfe\xff",
            b"\x80\x80\x80",
            b"\xe1\x80",
            b"\xf4\x8f\xbf\xbf",
            b"\n \t",
            // Long forms whose final continuation byte is missing: the only
            // thing that distinguishes "needs five continuations" from "needs
            // four" is a sequence that supplies exactly four.
            b"\xfd\xbf\xbf\xbf\xbf\x41",
            b"\xfc\x84\x80\x80\x80\x41",
            b"\xf8\x88\x80\x80\x41",
            b"\xf0\x90\x80\x41",
            b"\xe1\x80\x41",
            b"\xc3\x41",
        ];

        let mut seed = 0x5eed_1234_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let avx512 = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl");
        let avx2 = avx2_available();

        for mode in [WsMode { unicode: true, nbsp: false }, WsMode { unicode: true, nbsp: true }] {
            for _ in 0..3000 {
                let n = (rnd() % 400) as usize;
                let mut data = Vec::new();
                while data.len() < n {
                    data.extend_from_slice(pieces[(rnd() % 24) as usize]);
                }
                let expect = ws::count_scalar_unicode(&data, true, true, mode).3;

                if avx512 {
                    let got = unsafe { count_chars_only_avx512(&data, mode) };
                    assert_eq!(
                        expect, got,
                        "avx512 chars mismatch mode={mode:?} data={:02x?}", &data
                    );
                }
                if avx2 {
                    let got = unsafe { count_chars_only_avx2(&data, mode) };
                    assert_eq!(
                        expect, got,
                        "avx2 chars mismatch mode={mode:?} data={:02x?}", &data
                    );
                }
            }
        }

        // Every lead byte, paired with every interesting second byte, walked
        // across a full lane so each combination is tested at every alignment.
        for lead in [
            0xc1u8, 0xc2, 0xdf, 0xe0, 0xe1, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf4, 0xf7, 0xf8, 0xfb,
            0xfc, 0xfd, 0xfe, 0xff,
        ] {
            for second in [0x41u8, 0x7f, 0x80, 0x84, 0x88, 0x90, 0x9f, 0xa0, 0xbf, 0xc0] {
                for off in 0..70usize {
                    let mut data = vec![b'x'; off];
                    data.push(lead);
                    data.push(second);
                    data.extend_from_slice(&[0x80, 0x80, 0x80, 0x80]);
                    data.extend_from_slice(b"tail");
                    let expect = ws::count_scalar_unicode(&data, true, true, WsMode { unicode: true, nbsp: false }).3;
                    if avx512 {
                        let got = unsafe { count_chars_only_avx512(&data, WsMode { unicode: true, nbsp: false }) };
                        assert_eq!(
                            expect, got,
                            "avx512 chars mismatch lead={lead:02x} second={second:02x} off={off}"
                        );
                    }
                    if avx2 {
                        let got = unsafe { count_chars_only_avx2(&data, WsMode { unicode: true, nbsp: false }) };
                        assert_eq!(
                            expect, got,
                            "avx2 chars mismatch lead={lead:02x} second={second:02x} off={off}"
                        );
                    }
                }
            }
        }
    }

    /// The kernel decides whether a lane can hold multi-byte white space by
    /// looking at lead bytes and the bytes that follow them. Text hardly ever
    /// exercises that decision, but arbitrary bytes hit every combination,
    /// including lead bytes that are followed by the wrong byte and lead bytes
    /// that fall on a lane boundary.
    #[test]
    fn avx512_lines_words_match_scalar_on_arbitrary_bytes() {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")) {
            return;
        }
        let mut st = 0x243f6a8885a308d3u64;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let leads = [0xc2u8, 0xe1, 0xe2, 0xe3];
        for mode in [
            WsMode { unicode: true, nbsp: true },
            WsMode { unicode: true, nbsp: false },
        ] {
            for carry_in in [true, false] {
                for iter in 0..4000 {
                    let n = (rnd() % 300) as usize;
                    let mut data: Vec<u8> = (0..n).map(|_| (rnd() >> 24) as u8).collect();
                    // Bias some runs so lead bytes land near a lane boundary.
                    if iter % 3 == 0 && n > 70 {
                        let pos = 63 - (rnd() % 3) as usize;
                        data[pos] = leads[(rnd() % 4) as usize];
                    }
                    let (sl, sw, _sb, _sc, sk) =
                        ws::count_scalar_unicode(&data, carry_in, false, mode);
                    let (al, aw, _ab, _ac, ak) =
                        unsafe { count_lw_avx512(&data, carry_in, mode) };
                    assert_eq!(
                        (sl, sw, sk),
                        (al, aw, ak),
                        "avx512 lw mismatch on arbitrary bytes carry={carry_in} mode={mode:?} data={:02x?}",
                        &data
                    );
                }
            }
        }
    }
}
