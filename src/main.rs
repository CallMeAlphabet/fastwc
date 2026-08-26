#![feature(portable_simd)]

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

//! fastwc — a GNU-wc-compatible `wc` reimplementation, optimized for throughput.

use clihelp::{HelpPage, Row, Section};
use memmap2::Mmap;
use rayon::prelude::*;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, IsTerminal, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::process::ExitCode;

mod simd;
mod ws;
use simd::count_buf_mode;
use ws::WsMode;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TotalMode {
    Auto,
    Always,
    Only,
    Never,
}

struct Options {
    print_lines: bool,
    print_words: bool,
    print_chars: bool,
    print_bytes: bool,
    print_linelength: bool,
    debug: bool,
    end_of_opts: bool,
    total_mode: TotalMode,
    files_from: Option<String>,
    files: Vec<OsString>,
    ws_mode: WsMode,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            print_lines: false,
            print_words: false,
            print_chars: false,
            print_bytes: false,
            print_linelength: false,
            debug: false,
            end_of_opts: false,
            total_mode: TotalMode::Auto,
            files_from: None,
            files: Vec::new(),
            ws_mode: WsMode::from_env(),
        }
    }
}

fn row(short: &'static str, long: &'static str, desc: &'static str) -> Row {
    Row::new(short, long, desc)
}

fn row_val(
    short: &'static str,
    long: &'static str,
    placeholder: &'static str,
    desc: &'static str,
) -> Row {
    Row::with_value(short, long, placeholder, desc)
}

fn output_format_rows() -> Vec<Row> {
    vec![
        row("", "(default)", "canonical lines, words, and bytes"),
        row("-l", "--lines", "print the newline counts"),
        row("-w", "--words", "print the word counts"),
        row("-c", "--bytes", "print the byte counts"),
        row("-m", "--chars", "print the character counts"),
        row("-L", "--max-line-length", "print the maximum display width"),
    ]
}

fn total_files_rows() -> Vec<Row> {
    vec![
        row_val("", "--total", "<WHEN>", "when to print a line with total counts"),
        row_val("", "--files0-from", "<F>", "read input from the files specified by"),
    ]
}

fn misc_rows() -> Vec<Row> {
    vec![
        row("", "--debug", "indicate what line count acceleration is used"),
        row("", "--help", "show this help"),
        row("", "--version", "show version"),
    ]
}

fn sections() -> Vec<Section> {
    vec![
        Section { title: "OUTPUT FORMAT", note: None, rows: output_format_rows() },
        Section { title: "TOTAL & FILES", note: None, rows: total_files_rows() },
        Section { title: "MISC", note: None, rows: misc_rows() },
    ]
}

fn usage_err() -> ! {
    eprintln!("Try 'wc --help' for more information.");
    std::process::exit(1);
}

fn print_help() -> ! {
    print_help_body(io::stdout().is_terminal());
    std::process::exit(0);
}

fn print_help_body(on: bool) {
    let mut page = HelpPage::new("fastwc 0.1.0 - a high-performance, GNU-compatible wc reimplementation")
        .usage("fastwc [OPTION]... [FILE]...")
        .usage("fastwc [OPTION]... --files0-from=F")
        .usage("fastwc [OPTION]... -          read from stdin explicitly")
        .blurb(
            "Print newline, word, and byte counts for each FILE, and a total line if\n\
             more than one FILE is specified. A word is a nonempty sequence of non-white\n\
             space characters delimited by whitespace or by start/end of input.\n\n\
             With no FILE, or when FILE is -, read standard input.",
        );

    for section in sections() {
        page = page.section(section);
    }

    print!("{}", page.render(on));
}

fn print_version() -> ! {
    let on = io::stdout().is_terminal();
    let (bold, cyan, reset) = if on {
        ("\x1b[1m", "\x1b[36m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    println!("{bold}{cyan}fastwc{reset} 0.1.0 {bold}(GNU wc compatible){reset}");
    std::process::exit(0);
}

fn parse_args() -> Options {
    let mut opts = Options::default();
    let args: Vec<OsString> = env::args_os().collect();
    let mut i = 1;

    while i < args.len() {
        let arg = &args[i];
        let arg_bytes = arg.as_bytes();

        if opts.end_of_opts || (arg_bytes == b"-" && !opts.end_of_opts) || !arg_bytes.starts_with(b"-") {
            opts.files.push(arg.clone());
            i += 1;
            continue;
        }

        if arg_bytes == b"--" {
            opts.end_of_opts = true;
            i += 1;
            continue;
        }

        if arg_bytes.starts_with(b"--") {
            let s = arg.to_string_lossy();
            let (name, inline_val) = match s.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (s.as_ref(), None),
            };
            match name {
                "--bytes" => opts.print_bytes = true,
                "--chars" => opts.print_chars = true,
                "--lines" => opts.print_lines = true,
                "--words" => opts.print_words = true,
                "--max-line-length" => opts.print_linelength = true,
                "--debug" => opts.debug = true,
                "--help" => print_help(),
                "--version" => print_version(),
                "--files0-from" => {
                    let val = match inline_val {
                        Some(v) => v,
                        None => {
                            i += 1;
                            if i >= args.len() {
                                eprintln!("wc: option '--files0-from' requires an argument");
                                usage_err();
                            }
                            args[i].to_string_lossy().to_string()
                        }
                    };
                    opts.files_from = Some(val);
                }
                "--total" => {
                    let val = match inline_val {
                        Some(v) => v,
                        None => {
                            i += 1;
                            if i >= args.len() {
                                eprintln!("wc: option '--total' requires an argument");
                                usage_err();
                            }
                            args[i].to_string_lossy().to_string()
                        }
                    };
                    opts.total_mode = match val.as_str() {
                        "auto" => TotalMode::Auto,
                        "always" => TotalMode::Always,
                        "only" => TotalMode::Only,
                        "never" => TotalMode::Never,
                        other => {
                            eprintln!(
                                "wc: invalid argument '{}' for '--total'\nValid arguments are:\n  - 'auto'\n  - 'always'\n  - 'only'\n  - 'never'",
                                other
                            );
                            std::process::exit(1);
                        }
                    };
                }
                _ => {
                    eprintln!("wc: unrecognized option '{}'", name);
                    usage_err();
                }
            }
            i += 1;
            continue;
        }

        for (ci, ch) in String::from_utf8_lossy(&arg_bytes[1..]).chars().enumerate() {
            match ch {
                'c' => opts.print_bytes = true,
                'm' => opts.print_chars = true,
                'l' => opts.print_lines = true,
                'w' => opts.print_words = true,
                'L' => opts.print_linelength = true,
                'h' if ci == 0 && arg_bytes.len() == 2 => {}
                _ => {
                    eprintln!("wc: invalid option -- '{}'", ch);
                    usage_err();
                }
            }
        }
        i += 1;
    }

    if !(opts.print_lines || opts.print_words || opts.print_chars || opts.print_bytes || opts.print_linelength) {
        opts.print_lines = true;
        opts.print_words = true;
        opts.print_bytes = true;
    }

    opts
}

#[derive(Default, Clone, Copy)]
struct Counts {
    lines: u64,
    words: u64,
    chars: u64,
    bytes: u64,
    max_line_length: i64,
}

fn count_complicated(data: &[u8], want_chars: bool, carry_in: (bool, i64), mode: WsMode) -> (Counts, bool, i64) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let (mut in_word_ws, mut linepos) = carry_in;
    let mut max_len = 0i64;
    let avx2 = simd::avx2_available();

    let mut i = 0;
    while i < data.len() {
        let b = data[i];

        // Printable ASCII plus the plain space, the overwhelmingly common
        // case, needs no decoding and no table lookups: every byte is one
        // character of width one that cannot end a line. A whole run is
        // vectorised, leaving only the word transitions to count.
        if (0x20..0x7f).contains(&b) {
            let run = i;
            let (end, w, carry) = simd::simple_ascii_run(data, i, in_word_ws, avx2);
            i = end;
            linepos += (i - run) as i64;
            words += w;
            in_word_ws = carry;
            if want_chars {
                chars += (i - run) as u64;
            }
            continue;
        }

        // Multi-byte characters that no whitespace set can contain are
        // measured in bulk: they are all word material, so the run only has
        // to contribute its width and, at its start, one word transition.
        if mode.unicode && ((0xC3..=0xDF).contains(&b) || (0xE4..=0xEC).contains(&b) || b == 0xEE || b == 0xEF) {
            let (end, n, w) = ws::nonspace_run(data, i);
            if end > i {
                if in_word_ws {
                    words += 1;
                }
                in_word_ws = false;
                linepos += w;
                if want_chars {
                    chars += n;
                }
                i = end;
                continue;
            }
        }

        // In a unibyte locale each byte stands alone; otherwise decode, and
        // treat a malformed sequence as one byte that is neither a character
        // nor white space, exactly as GNU's mbrtoc32 error branch does.
        let (cp, char_len, valid) = if b < 0x80 {
            (b as u32, 1, true)
        } else if mode.unicode {
            ws::decode(data, i)
        } else {
            (b as u32, 1, true)
        };

        if !valid {
            if in_word_ws {
                words += 1;
            }
            in_word_ws = false;
            i += char_len;
            continue;
        }

        match cp {
            0x0a => {
                lines += 1;
                if linepos > max_len { max_len = linepos; }
                linepos = 0;
                in_word_ws = true;
            }
            0x0d | 0x0c => {
                if linepos > max_len { max_len = linepos; }
                linepos = 0;
                in_word_ws = true;
            }
            0x09 => {
                linepos += 8 - (linepos % 8);
                in_word_ws = true;
            }
            0x20 => {
                linepos += 1;
                in_word_ws = true;
            }
            0x0b => {
                in_word_ws = true;
            }
            _ => {
                linepos += ws::display_width(cp, mode.unicode);
                if ws::is_ws_char(cp, mode) {
                    in_word_ws = true;
                } else {
                    if in_word_ws { words += 1; }
                    in_word_ws = false;
                }
            }
        }
        if want_chars { chars += 1; }
        i += char_len;
    }

    (Counts { lines, words, chars, bytes: data.len() as u64, max_line_length: max_len }, in_word_ws, linepos)
}

fn trailing_ws(data: &[u8], mode: WsMode) -> bool {
    if data.is_empty() {
        return true;
    }
    let last = data[data.len() - 1];
    if last < 0x80 || !mode.unicode {
        return ws::is_ws_char(last as u32, mode);
    }
    let mut start = data.len() - 1;
    let mut steps = 0;
    while start > 0 && (data[start] & 0xC0) == 0x80 && steps < 3 {
        start -= 1;
        steps += 1;
    }
    let (cp, len, valid) = ws::decode(data, start);
    if !valid || start + len != data.len() {
        return false;
    }
    ws::is_ws_char(cp, mode)
}

fn count_parallel(
    data: &[u8],
    want_chars: bool,
    want_ws: bool,
    debug: bool,
    mode: WsMode,
) -> Counts {
    if data.is_empty() { return Counts::default(); }

    let nthreads = rayon::current_num_threads().max(1);
    let target_chunks = (nthreads * 4).max(1);
    let chunk_size = (data.len() / target_chunks).clamp(256 * 1024, 16 * 1024 * 1024);

    // A multi-byte space must not be cut in half across two threads.
    let chunks: Vec<&[u8]> = if mode.unicode {
        let mut bounds: Vec<usize> = Vec::new();
        let mut off = 0usize;
        while off < data.len() {
            let mut end = (off + chunk_size).min(data.len());
            while end < data.len() && (data[end] & 0xC0) == 0x80 {
                end += 1;
            }
            bounds.push(end);
            off = end;
        }
        let mut v = Vec::with_capacity(bounds.len());
        let mut start = 0usize;
        for &e in &bounds {
            v.push(&data[start..e]);
            start = e;
        }
        v
    } else {
        data.chunks(chunk_size).collect()
    };

    if debug {
        eprintln!(
            "wc: debug: {} chunk(s) of ~{} bytes across {} thread(s) (avx2={})",
            chunks.len(), chunk_size, nthreads, simd::avx2_available()
        );
    }

    if !want_ws {
        return chunks
            .par_iter()
            .map(|c| Counts {
                lines: 0,
                words: 0,
                chars: simd::count_chars_only(c, mode),
                bytes: c.len() as u64,
                max_line_length: 0,
            })
            .reduce(Counts::default, |a, b| Counts {
                lines: 0,
                words: 0,
                chars: a.chars + b.chars,
                bytes: a.bytes + b.bytes,
                max_line_length: 0,
            });
    }

    let boundary_last_ws: Vec<bool> = chunks
        .iter()
        .map(|c| trailing_ws(c, mode))
        .collect();
    let mut carries_in = vec![true; chunks.len()];
    for idx in 1..chunks.len() { carries_in[idx] = boundary_last_ws[idx - 1]; }

    chunks
        .par_iter()
        .zip(carries_in.par_iter())
        .map(|(c, &carry_in)| {
            let (lines, words, bytes, chars, _carry_out) = count_buf_mode(c, carry_in, want_chars, mode);
            Counts { lines, words, chars, bytes, max_line_length: 0 }
        })
        .reduce(Counts::default, |a, b| Counts {
            lines: a.lines + b.lines,
            words: a.words + b.words,
            chars: a.chars + b.chars,
            bytes: a.bytes + b.bytes,
            max_line_length: 0,
        })
}

/// `-L` counting spread across threads.
///
/// A newline is a hard reset: it zeroes the running column and leaves the
/// scanner between words. So a chunk that starts just after a newline needs
/// no state from its predecessor, and the results combine by summing the
/// counts and taking the maximum of the line lengths. Splitting on newlines
/// also keeps multi-byte characters intact for free.
///
/// Input with no newline in it at all yields a single chunk, which is simply
/// the serial path.
fn count_parallel_linelength(data: &[u8], want_chars: bool, mode: WsMode) -> Counts {
    if data.is_empty() {
        return Counts::default();
    }

    let nthreads = rayon::current_num_threads().max(1);
    let target_chunks = (nthreads * 4).max(1);
    let chunk_size = (data.len() / target_chunks).clamp(1024 * 1024, 32 * 1024 * 1024);

    let mut bounds: Vec<usize> = Vec::new();
    let mut off = 0usize;
    while off < data.len() {
        let want = (off + chunk_size).min(data.len());
        let end = if want == data.len() {
            data.len()
        } else {
            match data[want..].iter().position(|&b| b == b'\n') {
                Some(p) => want + p + 1,
                None => data.len(),
            }
        };
        bounds.push(end);
        off = end;
    }

    let mut chunks: Vec<&[u8]> = Vec::with_capacity(bounds.len());
    let mut start = 0usize;
    for &e in &bounds {
        chunks.push(&data[start..e]);
        start = e;
    }

    chunks
        .par_iter()
        .map(|c| {
            let (counts, _, linepos) = count_complicated(c, want_chars, (true, 0), mode);
            let mut counts = counts;
            if linepos > counts.max_line_length {
                counts.max_line_length = linepos;
            }
            counts
        })
        .reduce(Counts::default, |a, b| Counts {
            lines: a.lines + b.lines,
            words: a.words + b.words,
            chars: a.chars + b.chars,
            bytes: a.bytes + b.bytes,
            max_line_length: a.max_line_length.max(b.max_line_length),
        })
}

/// Number of bytes at the tail of a buffer that may be the start of a
/// multi-byte character continued in the next read. Never more than 3.
fn incomplete_tail(data: &[u8], mode: WsMode) -> usize {
    if !mode.unicode {
        return 0;
    }
    let n = data.len();
    let mut i = n;
    let mut steps = 0;
    while i > 0 && steps < 3 {
        i -= 1;
        steps += 1;
        let b = data[i];
        if (b & 0xC0) == 0x80 {
            continue;
        }
        let need = if b & 0xE0 == 0xC0 {
            2
        } else if b & 0xF0 == 0xE0 {
            3
        } else if b & 0xF8 == 0xF0 {
            4
        } else {
            return 0;
        };
        return if n - i < need { n - i } else { 0 };
    }
    0
}

fn count_stream(reader: &mut dyn Read, opts: &Options) -> CountOutcome {
    const BUF: usize = 1 << 20;
    // Room to prepend a partial character carried over from the last read.
    let mut buf = vec![0u8; BUF + 4];
    let want_chars = opts.print_chars;
    let mode = opts.ws_mode;

    let mut total = Counts::default();
    let mut carry_ws = true;
    let mut carry_pos = 0i64;
    let mut prev = 0usize;

    let mut read_err = None;
    loop {
        let n = match reader.read(&mut buf[prev..prev + BUF]) {
            Ok(n) => n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => {
                read_err = Some(e);
                break;
            }
        };
        if n == 0 {
            break;
        }
        let avail = prev + n;

        // Hold back a trailing partial sequence so it is decoded as one
        // character once the rest of it arrives.
        let hold = incomplete_tail(&buf[..avail], mode);
        let end = avail - hold;

        if opts.print_linelength {
            let (c, in_word_ws, linepos) =
                count_complicated(&buf[..end], want_chars, (carry_ws, carry_pos), mode);
            total.lines += c.lines;
            total.words += c.words;
            total.chars += c.chars;
            total.bytes += c.bytes;
            total.max_line_length = total.max_line_length.max(c.max_line_length);
            carry_ws = in_word_ws;
            carry_pos = linepos;
        } else {
            let (l, w, b, c, carry_out) = count_buf_mode(&buf[..end], carry_ws, want_chars, mode);
            total.lines += l;
            total.words += w;
            total.bytes += b;
            total.chars += c;
            carry_ws = carry_out;
        }

        buf.copy_within(end..avail, 0);
        prev = hold;
    }

    // Whatever is left is a genuine encoding error at end of input.
    if prev > 0 {
        let tail: Vec<u8> = buf[..prev].to_vec();
        if opts.print_linelength {
            let (c, _, linepos) =
                count_complicated(&tail, want_chars, (carry_ws, carry_pos), mode);
            total.lines += c.lines;
            total.words += c.words;
            total.chars += c.chars;
            total.bytes += c.bytes;
            total.max_line_length = total.max_line_length.max(c.max_line_length);
            carry_pos = linepos;
        } else {
            let (l, w, b, c, _) = count_buf_mode(&tail, carry_ws, want_chars, mode);
            total.lines += l;
            total.words += w;
            total.bytes += b;
            total.chars += c;
        }
    }

    if opts.print_linelength && carry_pos > total.max_line_length {
        total.max_line_length = carry_pos;
    }
    CountOutcome { counts: total, read_err }
}

/// Counting outcome. GNU distinguishes a file it could not *open* (no row is
/// printed) from one that failed while *reading* (the counts gathered so far
/// are still printed, then the error). `read_err` carries the second case.
struct CountOutcome {
    counts: Counts,
    read_err: Option<io::Error>,
}

fn count_path(path: Option<&OsString>, opts: &Options) -> io::Result<CountOutcome> {
    let is_stdin = path.is_none() || (path.map(|p| p.as_bytes() == b"-").unwrap_or(false) && !opts.end_of_opts);

    if is_stdin {
        let stdin = io::stdin();
        let mut lock = stdin.lock();
        return Ok(count_stream(&mut lock, opts));
    }

    let path = path.unwrap();
    let meta = fs::metadata(path)?;

    let only_bytes = opts.print_bytes
        && !opts.print_lines
        && !opts.print_words
        && !opts.print_chars
        && !opts.print_linelength;

    // Opening happens before the -c shortcut: an unreadable file is an error
    // even when its size alone would answer the question.
    let file = File::open(path)?;

    if only_bytes && meta.is_file() {
        return Ok(CountOutcome {
            counts: Counts { bytes: meta.len(), ..Counts::default() },
            read_err: None,
        });
    }

    // A directory opens fine; the failure only shows up on read, so GNU emits
    // a zero row followed by the diagnostic.
    if meta.is_dir() {
        return Ok(CountOutcome {
            counts: Counts::default(),
            read_err: Some(io::Error::from_raw_os_error(libc::EISDIR)),
        });
    }

    if meta.is_file() && meta.len() > 64 * 1024 {
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        unsafe {
            #[cfg(unix)]
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL);
        }
        let data: &[u8] = &mmap;

        if opts.print_linelength {
            let counts = count_parallel_linelength(data, opts.print_chars, opts.ws_mode);
            return Ok(CountOutcome { counts, read_err: None });
        }

        let want_ws = opts.print_lines || opts.print_words;
        let counts = count_parallel(data, opts.print_chars, want_ws, opts.debug, opts.ws_mode);
        return Ok(CountOutcome { counts, read_err: None });
    }

    let mut f = file;
    Ok(count_stream(&mut f, opts))
}

/// Split a NUL-separated name list. Zero-length names are preserved rather
/// than skipped: GNU diagnoses each one and reports its record number, so the
/// caller needs to see them.
fn read_files0_from(spec: &str) -> io::Result<Vec<OsString>> {
    let mut data = Vec::new();
    if spec == "-" {
        io::stdin().lock().read_to_end(&mut data)?;
    } else {
        File::open(spec)?.read_to_end(&mut data)?;
    }

    // Only a completely empty list has no names at all. Otherwise a trailing
    // separator merely closes the last name rather than opening a new one, so
    // a lone NUL still describes one (empty, and therefore invalid) name.
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.last() == Some(&0) {
        data.pop();
    }

    Ok(data
        .split(|&b| b == 0)
        .map(|part| OsString::from(std::ffi::OsStr::from_bytes(part)))
        .collect())
}


/// Column width for the numeric fields, following GNU's
/// `get_input_fstatus` / `compute_number_width`.
///
/// The width is decided from the *sizes* of the inputs before anything is
/// counted, not from the counts themselves. Every operand is stat'd and the
/// regular-file sizes are summed; the width is the digit count of that sum.
/// A non-regular input (pipe, terminal, character device) forces a minimum of
/// 7, which is why `wc -lwmcL < file` and `cat file | wc -lwmcL` disagree on
/// padding for the same bytes. Printing a single number needs no alignment,
/// so that case short-circuits to width 1 and skips the stat entirely.
fn compute_widths(opts: &Options, files: &[OsString], nflags: usize) -> [usize; 5] {
    if opts.total_mode == TotalMode::Only {
        return [1; 5];
    }

    // nfiles == 0 is the unknown-length --files0-from case.
    let nfiles = if opts.files_from.is_some() && files.is_empty() {
        0
    } else {
        files.len().max(1)
    };

    if nfiles == 0 || (nfiles == 1 && nflags == 1) {
        return [1; 5];
    }

    let mut minimum_width = 1usize;
    let mut regular_total = 0u64;
    let mut any = false;

    let mut note = |md: std::fs::Metadata| {
        any = true;
        if md.is_file() {
            regular_total = regular_total.saturating_add(md.len());
        } else {
            minimum_width = 7;
        }
    };

    if files.is_empty() {
        if let Ok(md) = fs::metadata("/dev/stdin") {
            note(md);
        } else {
            return [1; 5];
        }
    } else {
        for f in files {
            let md = if f.as_bytes() == b"-" && !opts.end_of_opts {
                fs::metadata("/dev/stdin")
            } else {
                fs::metadata(f)
            };
            if let Ok(md) = md {
                note(md);
            }
        }
    }

    if !any {
        return [1; 5];
    }

    let mut width = 1usize;
    while regular_total >= 10 {
        width += 1;
        regular_total /= 10;
    }
    [width.max(minimum_width); 5]
}

fn write_counts<W: Write>(
    out: &mut W,
    opts: &Options,
    c: &Counts,
    widths: &[usize; 5],
    name: Option<&str>,
) -> io::Result<()> {
    let mut first = true;
    macro_rules! field {
        ($idx:expr, $v:expr) => {{
            if first {
                write!(out, "{:>width$}", $v, width = widths[$idx])?;
                #[allow(unused_assignments)]
                {
                    first = false;
                }
            } else {
                write!(out, " {:>width$}", $v, width = widths[$idx])?;
            }
        }};
    }
    if opts.print_lines { field!(0, c.lines); }
    if opts.print_words { field!(1, c.words); }
    if opts.print_chars { field!(2, c.chars); }
    if opts.print_bytes { field!(3, c.bytes); }
    if opts.print_linelength { field!(4, c.max_line_length); }
    if let Some(n) = name {
        let escaped = if n.chars().any(|c| c.is_control()) {
            let mut s = String::new();
            s.push('\'');
            for ch in n.chars() {
                match ch {
                    '\n' => s.push_str("'$'\\n''"),
                    '\t' => s.push_str("'$'\\t''"),
                    '\r' => s.push_str("'$'\\r''"),
                    '\'' => s.push_str("'\\''"),
                    _ => s.push(ch),
                }
            }
            s.push('\'');
            s
        } else {
            n.to_string()
        };
        write!(out, " {escaped}")?;
    }
    writeln!(out)?;
    Ok(())
}

fn run() -> io::Result<bool> {
    let mut opts = parse_args();

    let mut ok = true;
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let file_list: Vec<OsString>;

    if let Some(spec) = opts.files_from.clone() {
        if !opts.files.is_empty() {
            eprintln!(
                "wc: extra operand {:?}\nfile operands cannot be combined with --files0-from",
                opts.files[0]
            );
            std::process::exit(1);
        }
        file_list = match read_files0_from(&spec) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("wc: cannot open '{spec}' for reading: {}", errmsg(&e));
                return Ok(false);
            }
        };
    } else if opts.files.is_empty() {
        file_list = vec![];
    } else {
        file_list = opts.files.clone();
    }

    opts.files = file_list.clone();

    let mut total = Counts::default();
    let nflags = opts.print_lines as usize
        + opts.print_words as usize
        + opts.print_chars as usize
        + opts.print_bytes as usize
        + opts.print_linelength as usize;
    let widths = compute_widths(&opts, &file_list, nflags);

    // An empty --files0-from list counts nothing at all, whereas an empty
    // command line means stdin.
    let empty_list = opts.files_from.is_some() && file_list.is_empty();

    // GNU decides on the total row from the number of operands it was given
    // (`argv_iter_n_args`), so a file that fails to open still counts towards
    // it: `wc f missing` prints a total row, and `wc missing missing` prints a
    // zero one.
    let nfiles_seen = if empty_list { 0 } else { file_list.len().max(1) as u64 };

    if empty_list {
        // Nothing to do.
    } else if file_list.is_empty() {
        match count_path(None, &opts) {
            Ok(out_c) => {
                total = out_c.counts;
                if opts.total_mode != TotalMode::Only {
                    write_counts(&mut out, &opts, &total, &widths, None)?;
                }
                if let Some(e) = out_c.read_err {
                    diag(&mut out, format_args!("-: {}", errmsg(&e)));
                    ok = false;
                }
            }
            Err(e) => {
                diag(&mut out, format_args!("-: {}", errmsg(&e)));
                ok = false;
            }
        }
    } else {
        for (i, f) in file_list.iter().enumerate() {
            let display = f.to_string_lossy().into_owned();

            // A zero-length name is diagnosed and skipped, never opened. With
            // --files0-from the record number is part of the message.
            if f.as_bytes().is_empty() {
                match &opts.files_from {
                    Some(spec) => diag(
                        &mut out,
                        format_args!("{spec}:{}: invalid zero-length file name", i + 1),
                    ),
                    None => diag(&mut out, format_args!("invalid zero-length file name")),
                }
                ok = false;
                continue;
            }

            // printf - | wc --files0-from=- cannot mean "read stdin twice".
            if opts.files_from.as_deref() == Some("-") && f.as_bytes() == b"-" {
                diag(
                    &mut out,
                    format_args!(
                        "when reading file names from stdin, no file name of '-' allowed"
                    ),
                );
                ok = false;
                continue;
            }

            match count_path(Some(f), &opts) {
                Ok(out_c) => {
                    let c = out_c.counts;
                    total.lines += c.lines;
                    total.words += c.words;
                    total.chars += c.chars;
                    total.bytes += c.bytes;
                    if c.max_line_length > total.max_line_length { total.max_line_length = c.max_line_length; }
                    if opts.total_mode != TotalMode::Only {
                        write_counts(&mut out, &opts, &c, &widths, Some(&display))?;
                    }
                    if let Some(e) = out_c.read_err {
                        diag(&mut out, format_args!("{display}: {}", errmsg(&e)));
                        ok = false;
                    }
                }
                Err(e) => {
                    diag(&mut out, format_args!("{display}: {}", errmsg(&e)));
                    ok = false;
                }
            }
        }
    }

    let print_total = match opts.total_mode {
        TotalMode::Never => false,
        TotalMode::Always | TotalMode::Only => true,
        TotalMode::Auto => nfiles_seen > 1,
    };

    if print_total {
        let name = if opts.total_mode == TotalMode::Only { None } else { Some("total") };
        write_counts(&mut out, &opts, &total, &widths, name)?;
    }

    out.flush()?;
    Ok(ok)
}

/// Print a diagnostic in GNU's order: stdout is flushed first so the message
/// lands after the rows already emitted rather than ahead of them.
fn diag(out: &mut dyn Write, msg: std::fmt::Arguments) {
    let _ = out.flush();
    eprintln!("wc: {msg}");
}

/// strerror text without Rust's trailing "(os error N)", matching GNU output.
fn errmsg(e: &io::Error) -> String {
    match e.raw_os_error() {
        Some(code) => {
            let s = unsafe { libc::strerror(code) };
            if s.is_null() {
                e.to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(s) }.to_string_lossy().into_owned()
            }
        }
        None => e.to_string(),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("wc: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GNU: WsMode = WsMode { unicode: true, nbsp: true };
    const CLOC: WsMode = WsMode { unicode: false, nbsp: true };

    fn complicated(data: &[u8], mode: WsMode) -> Counts {
        let (mut c, _, linepos) = count_complicated(data, true, (true, 0), mode);
        if linepos > c.max_line_length {
            c.max_line_length = linepos;
        }
        c
    }

    #[test]
    fn tab_advances_to_next_multiple_of_eight() {
        assert_eq!(complicated(b"a\tb", GNU).max_line_length, 9);
        assert_eq!(complicated(b"\t", GNU).max_line_length, 8);
        assert_eq!(complicated(b"abcdefgh\tx", GNU).max_line_length, 17);
    }

    #[test]
    fn carriage_return_and_formfeed_reset_linepos() {
        assert_eq!(complicated(b"abcdef\rxy", GNU).max_line_length, 6);
        assert_eq!(complicated(b"abcdef\x0cxy", GNU).max_line_length, 6);
        // Vertical tab keeps the column but ends the word.
        assert_eq!(complicated(b"abc\x0bdef", GNU).max_line_length, 6);
        assert_eq!(complicated(b"abc\x0bdef", GNU).words, 2);
    }

    #[test]
    fn nonprintable_bytes_have_no_width() {
        assert_eq!(complicated(b"\x01\x01\x01", GNU).max_line_length, 0);
        assert_eq!(complicated(b"ab\x01cd", GNU).max_line_length, 4);
    }

    #[test]
    fn unibyte_locale_treats_high_bytes_as_zero_width() {
        // LC_ALL=C: "caf\xc3\xa9" is five bytes, only three printable.
        let c = complicated(b"caf\xc3\xa9", CLOC);
        assert_eq!(c.max_line_length, 3);
        assert_eq!(c.chars, 5);
        assert_eq!(c.bytes, 5);
    }

    #[test]
    fn unibyte_locale_chars_equal_bytes() {
        let data = "caf\u{e9} \u{4e2d}\u{6587}".as_bytes();
        let (l, w, b, c, _) = count_buf_mode(data, true, true, CLOC);
        assert_eq!(c, b, "-m must equal -c in a unibyte locale");
        assert_eq!(l, 0);
        assert_eq!(w, 2);
    }

    #[test]
    fn malformed_sequences_are_bytes_not_characters() {
        let c = complicated(b"\xc0\xa0\xed\xa0\x80\xff", GNU);
        assert_eq!(c.bytes, 6);
        assert_eq!(c.chars, 0);
        assert_eq!(c.max_line_length, 0);
        // Encoding errors are word material, and these are all contiguous.
        assert_eq!(c.words, 1);
    }

    #[test]
    fn wide_and_combining_characters_use_wcwidth() {
        unsafe {
            libc::setlocale(libc::LC_ALL, c"C.utf8".as_ptr());
        }
        assert_eq!(complicated("\u{4e2d}\u{6587}".as_bytes(), GNU).max_line_length, 4);
        // U+2028 is a separator with wcwidth -1; it must not subtract.
        assert!(complicated("ab\u{2028}cd".as_bytes(), GNU).max_line_length >= 0);
    }

    #[test]
    fn incomplete_tail_holds_back_only_split_sequences() {
        // A whole character is never held back.
        assert_eq!(incomplete_tail("a\u{2003}".as_bytes(), GNU), 0);
        // A truncated one is, so the next read can complete it.
        assert_eq!(incomplete_tail(b"a\xe2\x80", GNU), 2);
        assert_eq!(incomplete_tail(b"a\xe2", GNU), 1);
        assert_eq!(incomplete_tail(b"a\xc2", GNU), 1);
        // Never in a unibyte locale, where bytes stand alone.
        assert_eq!(incomplete_tail(b"a\xe2\x80", CLOC), 0);
    }

    #[test]
    fn stream_matches_whole_buffer_across_read_boundary() {
        // A separator straddling the 1 MiB read boundary must still be one
        // character, at every possible split offset.
        for off in -3i64..3 {
            let pos = ((1usize << 20) as i64 + off) as usize;
            let mut data = vec![b'a'; pos];
            data.extend_from_slice("\u{2003}".as_bytes());
            data.extend_from_slice(&[b'b'; 100]);

            let opts = Options {
                print_words: true,
                print_chars: true,
                ws_mode: GNU,
                ..Options::default()
            };
            let streamed = count_stream(&mut &data[..], &opts).counts;
            let (_, words, _, chars, _) = count_buf_mode(&data, true, true, GNU);
            assert_eq!(streamed.words, words, "words at offset {off}");
            assert_eq!(streamed.chars, chars, "chars at offset {off}");
            assert_eq!(streamed.words, 2);
        }
    }

    /// Bug #8: the field width comes from the input *sizes*, discovered by
    /// stat before counting, never from the counts themselves.
    #[test]
    fn width_is_derived_from_file_sizes_not_from_counts() {
        let dir = std::env::temp_dir().join("fastwc_width_test");
        let _ = fs::create_dir_all(&dir);
        let big = dir.join("big");
        let small = dir.join("small");
        // 12345 bytes of newlines: 5-digit size, but only 5 digits of size and
        // a 5-digit line count that must not be what drives the width.
        fs::write(&big, vec![b'\n'; 12345]).unwrap();
        fs::write(&small, b"hi\n").unwrap();

        let files = vec![OsString::from(&big), OsString::from(&small)];
        let opts = Options {
            print_lines: true,
            print_words: true,
            files: files.clone(),
            ..Options::default()
        };

        // 12345 + 3 = 12348 bytes => 5 columns, for every field.
        assert_eq!(compute_widths(&opts, &files, 2), [5; 5]);

        // One file and one flag needs no alignment at all.
        let one = vec![OsString::from(&big)];
        let single = Options {
            print_lines: true,
            files: one.clone(),
            ..Options::default()
        };
        assert_eq!(compute_widths(&single, &one, 1), [1; 5]);

        // ...but one file with two flags is padded to the size width.
        let two_flags = Options {
            print_lines: true,
            print_words: true,
            files: one.clone(),
            ..Options::default()
        };
        assert_eq!(compute_widths(&two_flags, &one, 2), [5; 5]);

        // --total=only prints a single row, so it is never padded.
        let total_only = Options {
            print_lines: true,
            print_words: true,
            total_mode: TotalMode::Only,
            files: files.clone(),
            ..Options::default()
        };
        assert_eq!(compute_widths(&total_only, &files, 2), [1; 5]);

        // A missing operand is skipped, not fatal, and not counted.
        let missing = dir.join("does_not_exist");
        let with_missing = vec![OsString::from(&small), OsString::from(&missing)];
        let opts_missing = Options {
            print_lines: true,
            print_words: true,
            files: with_missing.clone(),
            ..Options::default()
        };
        assert_eq!(compute_widths(&opts_missing, &with_missing, 2), [1; 5]);

        // A non-regular input forces GNU's minimum width of 7.
        let with_dev = vec![OsString::from(&small), OsString::from("/dev/null")];
        let opts_dev = Options {
            print_lines: true,
            print_words: true,
            files: with_dev.clone(),
            ..Options::default()
        };
        assert_eq!(compute_widths(&opts_dev, &with_dev, 2), [7; 5]);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Bug #10: a NUL-separated list keeps zero-length entries so each can be
    /// diagnosed with its record number, and a trailing separator does not
    /// invent a final empty name.
    #[test]
    fn files0_list_preserves_empty_names_but_not_a_trailing_separator() {
        let dir = std::env::temp_dir().join("fastwc_files0_test");
        let _ = fs::create_dir_all(&dir);
        let list = dir.join("list");

        let read = |bytes: &[u8]| {
            fs::write(&list, bytes).unwrap();
            read_files0_from(list.to_str().unwrap()).unwrap()
        };

        assert_eq!(read(b"a\0b\0"), vec![OsString::from("a"), OsString::from("b")]);
        // No trailing separator is still two names.
        assert_eq!(read(b"a\0b"), vec![OsString::from("a"), OsString::from("b")]);
        // The empty middle entry survives, as record 2.
        assert_eq!(
            read(b"a\0\0b\0"),
            vec![OsString::from("a"), OsString::new(), OsString::from("b")]
        );
        // An empty list is empty, not one empty name.
        assert_eq!(read(b""), Vec::<OsString>::new());
        assert_eq!(read(b"\0"), vec![OsString::new()]);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Bug #9: a read failure still yields the counts gathered before it, so
    /// the row is printed and the diagnostic follows.
    #[test]
    fn read_errors_keep_the_counts_gathered_so_far() {
        struct FailsAfterOneRead(bool);
        impl Read for FailsAfterOneRead {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.0 {
                    return Err(io::Error::from_raw_os_error(libc::EIO));
                }
                self.0 = true;
                let data = b"one two\nthree\n";
                buf[..data.len()].copy_from_slice(data);
                Ok(data.len())
            }
        }

        let opts = Options { print_lines: true, print_words: true, ..Options::default() };
        let out = count_stream(&mut FailsAfterOneRead(false), &opts);
        assert_eq!(out.counts.lines, 2);
        assert_eq!(out.counts.words, 3);
        assert!(out.read_err.is_some(), "the error must still be reported");
    }
}
