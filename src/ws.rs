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

    pub unicode: bool,

    pub nbsp: bool,
}

impl WsMode {

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

pub const LEAD_C2: u8 = 0xC2;
pub const LEAD_E1: u8 = 0xE1;
pub const LEAD_E2: u8 = 0xE2;
pub const LEAD_E3: u8 = 0xE3;

pub const NBSP_BYTE: u8 = 0xA0;

#[inline(always)]
pub fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

#[inline]
pub fn is_ws_char(cp: u32, mode: WsMode) -> bool {
    if cp < 0x80 {
        return is_ascii_ws(cp as u8);
    }

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

#[inline(always)]
pub fn display_width(cp: u32, unicode: bool) -> i64 {

    if cp < 0x80 {
        return if (0x20..=0x7e).contains(&cp) { 1 } else { 0 };
    }
    if !unicode {
        return 0;
    }

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

#[inline]
pub fn decode(data: &[u8], i: usize) -> (u32, usize, bool) {
    let b = data[i];
    if b < 0x80 {
        return (b as u32, 1, true);
    }

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
        let (cp, len, valid) = if b < 0x80 {
            (b as u32, 1, true)
        } else if !want_chars && !matches!(b, LEAD_C2 | LEAD_E1 | LEAD_E2 | LEAD_E3) {
            (0xFFFD, 1, false)
        } else {
            decode(data, i)
        };

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

        assert_eq!(decode(&[0xfb, 0xb8, 0x96, 0xb3, 0x93], 0), (0x3E16CD3, 5, true));
        assert_eq!(decode(&[0xf8, 0x88, 0x80, 0x80, 0x80], 0), (0x20_0000, 5, true));
        assert_eq!(decode(&[0xfc, 0x84, 0x80, 0x80, 0x80, 0x80], 0), (0x400_0000, 6, true));
        assert_eq!(decode(&[0xfd, 0xbf, 0xbf, 0xbf, 0xbf, 0xbf], 0), (0x7FFF_FFFF, 6, true));

        assert_eq!(decode(&[0xf4, 0x90, 0x80, 0x80], 0), (0x11_0000, 4, true));
        assert_eq!(decode(&[0xf7, 0xbf, 0xbf, 0xbf], 0), (0x1F_FFFF, 4, true));

        assert!(!decode(&[0xf8, 0x87, 0xbf, 0xbf, 0xbf], 0).2);
        assert!(!decode(&[0xfc, 0x83, 0xbf, 0xbf, 0xbf, 0xbf], 0).2);
        assert!(!decode(&[0xed, 0xa0, 0x80], 0).2);
        assert!(!decode(&[0xc0, 0xa0], 0).2);
        assert!(!decode(&[0xc1, 0xbf], 0).2);
        assert!(!decode(&[0xe0, 0x80, 0xa0], 0).2);
        assert!(!decode(&[0xfe, 0x80, 0x80], 0).2);
        assert!(!decode(&[0xff], 0).2);

        assert!(!decode(&[0xfb, 0xb8], 0).2);
    }

    #[test]
    fn glibc_iswspace_set() {

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

        unsafe {
            libc::setlocale(libc::LC_ALL, c"C.utf8".as_ptr());
        }
        assert_eq!(display_width(0x01, true), 0);
        assert_eq!(display_width(0x7f, true), 0);
        assert_eq!(display_width(b'a' as u32, true), 1);
        assert_eq!(display_width(0x4e2d, true), 2);

        assert_eq!(display_width(0x2028, true), 0);
        assert_eq!(display_width(0x200b, true), 0);

        assert_eq!(display_width(0xc3, false), 0);
        assert_eq!(display_width(0xa0, false), 0);
        assert_eq!(display_width(b'a' as u32, false), 1);
    }
}
