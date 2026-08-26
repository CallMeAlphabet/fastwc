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

//! Locale-aware character classification for word counting.
//!
//! POSIX defines a word as a string delimited by white space and defers the
//! definition of white space to `LC_CTYPE` (`iswspace`). In a UTF-8 locale
//! glibc's `iswspace` is true for exactly:
//!
//!     U+0009..U+000D, U+0020, U+1680, U+2000..U+2006,
//!     U+2008..U+200A, U+2028..U+2029, U+205F, U+3000
//!
//! GNU wc additionally treats U+00A0, U+2007, U+202F and U+2060 as delimiters
//! unless POSIXLY_CORRECT is set; we reproduce that for drop-in compatibility.
//! In a unibyte locale GNU derives the same table through `btoc32`, which maps
//! bytes as ISO-8859-1, so byte 0xA0 becomes a delimiter there too.
//!
//! Every non-ASCII codepoint above encodes to UTF-8 with lead byte 0xE1, 0xE2
//! or 0xE3 -- plus 0xC2 for U+00A0 -- and none of those can appear as a
//! continuation byte. A block containing none of them provably contains no
//! non-ASCII whitespace, which is what lets the SIMD kernel keep its ASCII
//! fast path.

use std::sync::atomic::{AtomicI8, Ordering};

unsafe extern "C" {
    fn wcwidth(c: libc::wchar_t) -> libc::c_int;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn __ctype_get_mb_cur_max() -> libc::size_t;
}

fn mb_cur_max() -> libc::size_t {
    #[cfg(target_os = "linux")]
    unsafe {
        __ctype_get_mb_cur_max()
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    unsafe {
        unsafe extern "C" {
            fn ___mb_cur_max() -> libc::c_int;
        }
        ___mb_cur_max() as libc::size_t
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
    {
        unsafe { libc::MB_CUR_MAX as libc::size_t }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WsMode {
    /// False in unibyte locales such as LC_ALL=C, where each byte is a
    /// character and multi-byte sequences are never decoded.
    pub unicode: bool,
    /// GNU extension, disabled by POSIXLY_CORRECT.
    pub nbsp: bool,
}

impl WsMode {
    /// Determine the mode the way GNU wc does: apply the environment's locale
    /// with `setlocale`, then branch on `MB_CUR_MAX`.
    ///
    /// Asking libc rather than parsing `LC_ALL` matters, because a locale the
    /// system does not have installed silently falls back to C. Trusting the
    /// variable would make us decode UTF-8 where real wc counts bytes.
    pub fn from_env() -> Self {
        let mb_cur_max = unsafe {
            libc::setlocale(libc::LC_ALL, c"".as_ptr());
            mb_cur_max()
        };
        WsMode {
            unicode: mb_cur_max > 1,
            nbsp: std::env::var_os("POSIXLY_CORRECT").is_none(),
        }
    }
}

pub const LEAD_C2: u8 = 0xC2; // U+00A0
pub const LEAD_E1: u8 = 0xE1; // U+1680
pub const LEAD_E2: u8 = 0xE2; // U+2000..U+206F block
pub const LEAD_E3: u8 = 0xE3; // U+3000

/// Byte 0xA0, a delimiter in unibyte locales unless POSIXLY_CORRECT.
pub const NBSP_BYTE: u8 = 0xA0;

/// The six bytes glibc's `isspace` accepts in any locale.
#[inline(always)]
pub fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Is `cp` whitespace for word-splitting purposes under `mode`?
///
/// In a unibyte locale `cp` is a raw byte, which reaches the same answer as
/// GNU's `isspace(i) || maybe_c32isnbspace(btoc32(i))` table.
#[inline]
pub fn is_ws_char(cp: u32, mode: WsMode) -> bool {
    if cp < 0x80 {
        return is_ascii_ws(cp as u8);
    }
    // U+3000 is the largest delimiter in either set, so everything above it
    // -- most of CJK, and every supplementary plane -- settles in one compare
    // instead of walking the match arms below.
    if cp > 0x3000 {
        return false;
    }
    if mode.nbsp && matches!(cp, 0x00A0 | 0x2007 | 0x202F | 0x2060) {
        return true;
    }
    if !mode.unicode {
        return false;
    }
    matches!(cp,
        0x1680
        | 0x2000..=0x2006
        | 0x2008..=0x200A
        | 0x2028..=0x2029
        | 0x205F
        | 0x3000
    )
}

/// Column width of `cp` for `-L`, mirroring GNU's `isprint` / `c32width`.
///
/// Unibyte locales use the C-locale `isprint`, which excludes every byte above
/// 0x7E. Otherwise defer to the platform's `wcwidth`: GNU adds the width only
/// when it is positive, so non-printable and incomplete characters (-1) and
/// combining marks (0) contribute nothing. The unicode-width crate is not a
/// substitute here, as it disagrees with glibc on the separator characters
/// this program has to classify.
#[inline(always)]
pub fn display_width(cp: u32, unicode: bool) -> i64 {
    // ASCII has the same width in every locale, so the common case never
    // pays for a call into libc.
    if cp < 0x80 {
        return if (0x20..=0x7e).contains(&cp) { 1 } else { 0 };
    }
    if !unicode {
        return 0;
    }
    // wcwidth is a locale-sensitive libc call and dominates -L on non-Latin
    // text, so results for the Basic Multilingual Plane are memoised. The
    // locale cannot change while counting, which makes the cache safe.
    if cp < 0x10000 {
        let idx = cp as usize;
        let cached = WIDTH_CACHE[idx].load(Ordering::Relaxed);
        if cached != WIDTH_UNKNOWN {
            return cached as i64;
        }
        let w = unsafe { wcwidth(cp as libc::wchar_t) };
        let w = if w > 0 { w as i8 } else { 0 };
        WIDTH_CACHE[idx].store(w, Ordering::Relaxed);
        return w as i64;
    }
    let w = unsafe { wcwidth(cp as libc::wchar_t) };
    if w > 0 {
        w as i64
    } else {
        0
    }
}

const WIDTH_UNKNOWN: i8 = -1;

static WIDTH_CACHE: [AtomicI8; 0x10000] = {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: AtomicI8 = AtomicI8::new(WIDTH_UNKNOWN);
    [INIT; 0x10000]
};

/// Decode one UTF-8 scalar at `data[i]`, returning (codepoint, length, valid).
///
/// Invalid sequences consume exactly one byte and report `valid = false`. GNU
/// counts such a byte as a byte but not as a character, and treats it as word
/// material rather than white space.
#[inline]
pub fn decode(data: &[u8], i: usize) -> (u32, usize, bool) {
    let b = data[i];
    if b < 0x80 {
        return (b as u32, 1, true);
    }
    // glibc's UTF-8 decoder still accepts the pre-RFC-3629 encoding: sequences
    // of up to six bytes, and every scalar value up to U+7FFFFFFF rather than
    // stopping at U+10FFFF. It rejects only overlong forms, surrogates and the
    // lead bytes 0xC0, 0xC1, 0xFE and 0xFF. Matching that exactly is what keeps
    // -m and -L in step with wc on arbitrary binary input.
    // Two- and three-byte sequences cover every non-ASCII character in
    // ordinary text, so they are decoded straight-line rather than through
    // the general loop below.
    if b < 0xE0 {
        if b < 0xC2 {
            return (0xFFFD, 1, false);
        }
        if i + 2 > data.len() {
            return (0xFFFD, 1, false);
        }
        let c1 = data[i + 1];
        if c1 & 0xC0 != 0x80 {
            return (0xFFFD, 1, false);
        }
        return ((((b as u32) & 0x1F) << 6) | (c1 as u32 & 0x3F), 2, true);
    }
    if b < 0xF0 {
        if i + 3 > data.len() {
            return (0xFFFD, 1, false);
        }
        let c1 = data[i + 1];
        let c2 = data[i + 2];
        if (c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 {
            return (0xFFFD, 1, false);
        }
        let cp = (((b as u32) & 0x0F) << 12) | ((c1 as u32 & 0x3F) << 6) | (c2 as u32 & 0x3F);
        if cp < 0x800 || (0xD800..0xE000).contains(&cp) {
            return (0xFFFD, 1, false);
        }
        return (cp, 3, true);
    }

    let (len, min) = match b {
        0xC2..=0xDF => (2usize, 0x80u32),
        0xE0..=0xEF => (3, 0x800),
        0xF0..=0xF7 => (4, 0x10000),
        0xF8..=0xFB => (5, 0x20_0000),
        0xFC..=0xFD => (6, 0x400_0000),
        _ => return (0xFFFD, 1, false),
    };
    if i + len > data.len() {
        return (0xFFFD, 1, false);
    }
    let mut cp = (b as u32) & (0x7F >> len);
    for k in 1..len {
        let c = data[i + k];
        if c & 0xC0 != 0x80 {
            return (0xFFFD, 1, false);
        }
        cp = (cp << 6) | (c as u32 & 0x3F);
    }
    if cp < min || (0xD800..0xE000).contains(&cp) {
        return (0xFFFD, 1, false);
    }
    (cp, len, true)
}

/// Measure a run of multi-byte characters that cannot be white space.
///
/// The whitespace sets contain exactly one two-byte character (U+00A0, lead
/// 0xC2) and a handful of three-byte ones, all with lead 0xE1, 0xE2 or 0xE3.
/// So a sequence led by 0xC3..=0xDF, 0xE4..=0xEC, 0xEE or 0xEF is guaranteed
/// non-space, and it is guaranteed well-formed too: those ranges cannot be
/// overlong and cannot encode a surrogate. Such a character therefore needs
/// no classification at all, only its width.
///
/// Collapsing a whole run into one pass takes the per-character `match`, the
/// whitespace test and the re-validation out of the loop, which is what makes
/// -L expensive on CJK and Cyrillic text.
///
/// Returns the end offset, how many characters were consumed and the total
/// column width they occupy.
pub fn nonspace_run(data: &[u8], start: usize) -> (usize, u64, i64) {
    let mut i = start;
    let mut chars = 0u64;
    let mut width = 0i64;
    let len = data.len();

    while i < len {
        let b = data[i];
        let cp = if (0xC3..=0xDF).contains(&b) {
            if i + 2 > len || data[i + 1] & 0xC0 != 0x80 {
                break;
            }
            let cp = (((b as u32) & 0x1F) << 6) | (data[i + 1] as u32 & 0x3F);
            i += 2;
            cp
        } else if (0xE4..=0xEC).contains(&b) || b == 0xEE || b == 0xEF {
            if i + 3 > len || data[i + 1] & 0xC0 != 0x80 || data[i + 2] & 0xC0 != 0x80 {
                break;
            }
            let cp = (((b as u32) & 0x0F) << 12)
                | ((data[i + 1] as u32 & 0x3F) << 6)
                | (data[i + 2] as u32 & 0x3F);
            i += 3;
            cp
        } else {
            break;
        };
        chars += 1;
        width += display_width(cp, true);
    }

    (i, chars, width)
}

/// Returns (lines, words, bytes, chars, carry_out); `carry_out` is true if
/// the buffer ended inside whitespace. Multibyte locales only.
pub fn count_scalar_unicode(
    data: &[u8],
    carry_in: bool,
    want_chars: bool,
    mode: WsMode,
) -> (u64, u64, u64, u64, bool) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let mut prev_ws = carry_in;

    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        let (cp, len, valid) = if b < 0x80 { (b as u32, 1, true) } else { decode(data, i) };

        if b == b'\n' {
            lines += 1;
        }
        let ws = valid && is_ws_char(cp, mode);
        if !ws && prev_ws {
            words += 1;
        }
        prev_ws = ws;
        if want_chars && valid {
            chars += 1;
        }
        i += len;
    }

    (lines, words, data.len() as u64, chars, prev_ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GNU: WsMode = WsMode { unicode: true, nbsp: true };
    const POSIX: WsMode = WsMode { unicode: true, nbsp: false };
    const CLOC: WsMode = WsMode { unicode: false, nbsp: true };
    const CLOC_POSIX: WsMode = WsMode { unicode: false, nbsp: false };

    #[test]
    fn decode_accepts_the_legacy_forms_glibc_accepts() {
        // glibc's UTF-8 decoder predates RFC 3629 and still takes five- and
        // six-byte sequences up to U+7FFFFFFF; wc counts them as one character
        // each, so the decoder has to agree.
        assert_eq!(decode(&[0xfb, 0xb8, 0x96, 0xb3, 0x93], 0), (0x3E16CD3, 5, true));
        assert_eq!(decode(&[0xf8, 0x88, 0x80, 0x80, 0x80], 0), (0x20_0000, 5, true));
        assert_eq!(decode(&[0xfc, 0x84, 0x80, 0x80, 0x80, 0x80], 0), (0x400_0000, 6, true));
        assert_eq!(decode(&[0xfd, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf], 0), (0x7FFF_FFFF, 6, true));
        // Beyond U+10FFFF but reachable in four bytes: still one character.
        assert_eq!(decode(&[0xf4, 0x90, 0x80, 0x80], 0), (0x11_0000, 4, true));
        assert_eq!(decode(&[0xf7, 0xbf, 0xbf, 0xbf], 0), (0x1F_FFFF, 4, true));

        // Still rejected: overlong forms, surrogates and the impossible leads.
        assert!(!decode(&[0xf8, 0x87, 0xbf, 0xbf, 0xbf], 0).2);
        assert!(!decode(&[0xfc, 0x83, 0xbf, 0xbf, 0xbf, 0xbf], 0).2);
        assert!(!decode(&[0xed, 0xa0, 0x80], 0).2);
        assert!(!decode(&[0xc0, 0xa0], 0).2);
        assert!(!decode(&[0xc1, 0xbf], 0).2);
        assert!(!decode(&[0xe0, 0x80, 0xa0], 0).2);
        assert!(!decode(&[0xfe, 0x80, 0x80], 0).2);
        assert!(!decode(&[0xff], 0).2);
        // Truncated tails are invalid, not partial characters.
        assert!(!decode(&[0xfb, 0xb8], 0).2);
    }

    #[test]
    fn glibc_iswspace_set() {
        // Dumped from glibc via iswspace() in C.utf8.
        let spaces = [
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x1680, 0x2000, 0x2001, 0x2002,
            0x2003, 0x2004, 0x2005, 0x2006, 0x2008, 0x2009, 0x200A, 0x2028,
            0x2029, 0x205F, 0x3000,
        ];
        for cp in spaces {
            assert!(is_ws_char(cp, POSIX), "U+{cp:04X} should be space");
        }
        for cp in [0x00A0u32, 0x2007, 0x202F, 0x2060, 0x200B, 0x180E, 0x0085, 0xFEFF, 0x2011] {
            assert!(!is_ws_char(cp, POSIX), "U+{cp:04X} should not be space");
        }
    }

    #[test]
    fn gnu_adds_nonbreaking() {
        for cp in [0x00A0u32, 0x2007, 0x202F, 0x2060] {
            assert!(is_ws_char(cp, GNU));
            assert!(!is_ws_char(cp, POSIX));
        }
        // ZWSP and friends are never delimiters, even for GNU.
        for cp in [0x200Bu32, 0x180E, 0x0085, 0xFEFF] {
            assert!(!is_ws_char(cp, GNU));
        }
    }

    #[test]
    fn unibyte_splits_on_byte_a0_only() {
        assert!(is_ws_char(NBSP_BYTE as u32, CLOC));
        assert!(!is_ws_char(NBSP_BYTE as u32, CLOC_POSIX));
        for cp in [0x80u32, 0xC2, 0xE2, 0xFF] {
            assert!(!is_ws_char(cp, CLOC));
        }
    }

    #[test]
    fn counts_words_across_unicode_space() {
        let s = "a\u{2003}b".as_bytes();
        let (_, w, _, _, _) = count_scalar_unicode(s, true, false, POSIX);
        assert_eq!(w, 2);
    }

    #[test]
    fn invalid_utf8_is_word_material() {
        let (_, w, _, _, _) = count_scalar_unicode(b"a\xffb", true, false, GNU);
        assert_eq!(w, 1);
    }

    #[test]
    fn invalid_bytes_are_bytes_but_not_characters() {
        for bad in [&b"\xc0\xa0"[..], b"\xed\xa0\x80", b"\xe2\x80", b"\xff", b"\xc2"] {
            let (_, _, _, c, _) = count_scalar_unicode(bad, true, true, GNU);
            assert_eq!(c, 0, "{bad:?} should contribute no characters");
        }
        let (_, _, _, c, _) = count_scalar_unicode("caf\u{e9}".as_bytes(), true, true, GNU);
        assert_eq!(c, 4);
    }

    #[test]
    fn decode_rejects_overlong_and_surrogates() {
        assert_eq!(decode(b"\xc0\xa0", 0), (0xFFFD, 1, false));
        assert_eq!(decode(b"\xed\xa0\x80", 0), (0xFFFD, 1, false));
        assert_eq!(decode("\u{3000}".as_bytes(), 0), (0x3000, 3, true));
    }

    #[test]
    fn width_matches_isprint_rules() {
        // wcwidth is locale-sensitive; ask for a multibyte locale first.
        unsafe {
            libc::setlocale(libc::LC_ALL, c"C.utf8".as_ptr());
        }
        assert_eq!(display_width(0x01, true), 0);
        assert_eq!(display_width(0x7f, true), 0);
        assert_eq!(display_width(b'a' as u32, true), 1);
        assert_eq!(display_width(0x4e2d, true), 2);
        // Negative wcwidth (non-printable) must add nothing, not underflow.
        assert_eq!(display_width(0x2028, true), 0);
        assert_eq!(display_width(0x200b, true), 0);
        // Unibyte: nothing above 0x7E is printable in the C locale.
        assert_eq!(display_width(0xc3, false), 0);
        assert_eq!(display_width(0xa0, false), 0);
        assert_eq!(display_width(b'a' as u32, false), 1);
    }
}
