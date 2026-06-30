//! SIMD-accelerated counting kernel for fastwc.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[u8], carry_in: bool, want_chars: bool) -> (u64, u64, u64, u64, bool) {
        count_buf_scalar(data, carry_in, want_chars)
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
}
