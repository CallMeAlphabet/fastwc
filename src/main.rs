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
use unicode_width::UnicodeWidthChar;

mod simd;
use simd::count_buf;

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

#[inline]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn count_complicated(data: &[u8], want_chars: bool, carry_in: (bool, i64)) -> (Counts, bool, i64) {
    let mut lines = 0u64;
    let mut words = 0u64;
    let mut chars = 0u64;
    let (mut in_word_ws, mut linepos) = carry_in;
    let mut max_len = 0i64;

    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        
        let (char_len, width) = if b < 0x80 {
            (1, 1)
        } else if b & 0xE0 == 0xC0 {
            if i + 1 < data.len() && data[i+1] & 0xC0 == 0x80 {
                let c = ((b as u32 & 0x1F) << 6) | (data[i+1] as u32 & 0x3F);
                (2, char::from_u32(c).map(|c| c.width().unwrap_or(0) as i64).unwrap_or(0))
            } else {
                (1, 0)
            }
        } else if b & 0xF0 == 0xE0 {
            if i + 2 < data.len() && data[i+1] & 0xC0 == 0x80 && data[i+2] & 0xC0 == 0x80 {
                let c = ((b as u32 & 0x0F) << 12) | ((data[i+1] as u32 & 0x3F) << 6) | (data[i+2] as u32 & 0x3F);
                (3, char::from_u32(c).map(|c| c.width().unwrap_or(0) as i64).unwrap_or(0))
            } else {
                (1, 0)
            }
        } else if b & 0xF8 == 0xF0 {
            if i + 3 < data.len() && data[i+1] & 0xC0 == 0x80 && data[i+2] & 0xC0 == 0x80 && data[i+3] & 0xC0 == 0x80 {
                let c = ((b as u32 & 0x07) << 18) | ((data[i+1] as u32 & 0x3F) << 12) | ((data[i+2] as u32 & 0x3F) << 6) | (data[i+3] as u32 & 0x3F);
                (4, char::from_u32(c).map(|c| c.width().unwrap_or(0) as i64).unwrap_or(0))
            } else {
                (1, 0)
            }
        } else {
            (1, 0)
        };

        match b {
            b'\n' => {
                lines += 1;
                if linepos > max_len { max_len = linepos; }
                linepos = 0;
                in_word_ws = true;
            }
            b'\r' | 0x0c => {
                if linepos > max_len { max_len = linepos; }
                linepos = 0;
                in_word_ws = true;
            }
            b'\t' => {
                linepos += 8 - (linepos % 8);
                in_word_ws = true;
            }
            b' ' | 0x0b => {
                if b == b' ' { linepos += 1; }
                in_word_ws = true;
            }
            _ => {
                linepos += width;
                if in_word_ws { words += 1; }
                in_word_ws = false;
            }
        }
        if want_chars && (b & 0xC0) != 0x80 { chars += 1; }
        i += char_len;
    }

    (Counts { lines, words, chars, bytes: data.len() as u64, max_line_length: max_len }, in_word_ws, linepos)
}

fn count_parallel(data: &[u8], want_chars: bool, debug: bool) -> Counts {
    if data.is_empty() { return Counts::default(); }

    let nthreads = rayon::current_num_threads().max(1);
    let target_chunks = (nthreads * 4).max(1);
    let chunk_size = (data.len() / target_chunks).clamp(256 * 1024, 16 * 1024 * 1024);
    let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();

    if debug {
        eprintln!(
            "wc: debug: {} chunk(s) of ~{} bytes across {} thread(s) (avx2={})",
            chunks.len(), chunk_size, nthreads, simd::avx2_available()
        );
    }

    let boundary_last_ws: Vec<bool> = chunks.iter().map(|c| is_ws_byte(c[c.len() - 1])).collect();
    let mut carries_in = vec![true; chunks.len()];
    for idx in 1..chunks.len() { carries_in[idx] = boundary_last_ws[idx - 1]; }

    chunks
        .par_iter()
        .zip(carries_in.par_iter())
        .map(|(c, &carry_in)| {
            let (lines, words, bytes, chars, _carry_out) = count_buf(c, carry_in, want_chars);
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

fn count_stream(reader: &mut dyn Read, opts: &Options) -> io::Result<Counts> {
    const BUF: usize = 1 << 20;
    let mut buf = vec![0u8; BUF];
    let want_chars = opts.print_chars;

    if opts.print_linelength {
        let mut total = Counts::default();
        let mut carry = (true, 0i64);
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 { break; }
            let (c, in_word_ws, linepos) = count_complicated(&buf[..n], want_chars, carry);
            total.lines += c.lines;
            total.words += c.words;
            total.chars += c.chars;
            total.bytes += c.bytes;
            total.max_line_length = total.max_line_length.max(c.max_line_length);
            carry = (in_word_ws, linepos);
        }
        if carry.1 > total.max_line_length { total.max_line_length = carry.1; }
        Ok(total)
    } else {
        let mut total = Counts::default();
        let mut carry = true;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 { break; }
            let (l, w, b, c, carry_out) = count_buf(&buf[..n], carry, want_chars);
            total.lines += l;
            total.words += w;
            total.bytes += b;
            total.chars += c;
            carry = carry_out;
        }
        Ok(total)
    }
}

struct FileResult {
    counts: Counts,
    display_name: Option<String>,
}

fn count_path(path: Option<&OsString>, opts: &Options) -> io::Result<Counts> {
    let is_stdin = path.is_none() || (path.map(|p| p.as_bytes() == b"-").unwrap_or(false) && !opts.end_of_opts);

    if is_stdin {
        let stdin = io::stdin();
        let mut lock = stdin.lock();
        return count_stream(&mut lock, opts);
    }

    let path = path.unwrap();
    let meta = fs::metadata(path)?;

    let only_bytes = opts.print_bytes
        && !opts.print_lines
        && !opts.print_words
        && !opts.print_chars
        && !opts.print_linelength;

    if only_bytes && meta.is_file() {
        return Ok(Counts { bytes: meta.len(), ..Counts::default() });
    }

    let file = File::open(path)?;

    if meta.is_file() && meta.len() > 64 * 1024 {
        let mmap = unsafe { Mmap::map(&file)? };
        #[cfg(unix)]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL);
        }
        let data: &[u8] = &mmap;

        if opts.print_linelength {
            let (counts, _, linepos) = count_complicated(data, opts.print_chars, (true, 0));
            let mut counts = counts;
            if linepos > counts.max_line_length { counts.max_line_length = linepos; }
            return Ok(counts);
        }

        return Ok(count_parallel(data, opts.print_chars, opts.debug));
    }

    let mut f = file;
    count_stream(&mut f, opts)
}

fn read_files0_from(spec: &str) -> io::Result<Vec<OsString>> {
    let mut data = Vec::new();
    if spec == "-" {
        io::stdin().lock().read_to_end(&mut data)?;
    } else {
        File::open(spec)?.read_to_end(&mut data)?;
    }

    let mut out = Vec::new();
    for part in data.split(|&b| b == 0) {
        if part.is_empty() { continue; }
        out.push(OsString::from(std::ffi::OsStr::from_bytes(part)));
    }
    Ok(out)
}

fn update_width(val: u64) -> usize {
    let mut width = 1;
    let mut v = val;
    while v >= 10 {
        width += 1;
        v /= 10;
    }
    width
}

fn compute_widths(opts: &Options, results: &[FileResult], total: &Counts, _used_files0: bool) -> [usize; 5] {
    if opts.total_mode == TotalMode::Only || opts.files_from.as_deref() == Some("-") {
        return [1; 5];
    }

    let nflags = opts.print_lines as usize
        + opts.print_words as usize
        + opts.print_chars as usize
        + opts.print_bytes as usize
        + opts.print_linelength as usize;

    let print_total = match opts.total_mode {
        TotalMode::Never => false,
        TotalMode::Always | TotalMode::Only => true,
        TotalMode::Auto => results.len() > 1,
    };

    if nflags == 1 && results.len() == 1 && !print_total {
        return [1; 5];
    }

    let mut max_w = 1;
    let mut consider = |c: &Counts| {
        max_w = max_w.max(update_width(c.lines));
        max_w = max_w.max(update_width(c.words));
        max_w = max_w.max(update_width(c.chars));
        max_w = max_w.max(update_width(c.bytes));
    };

    for r in results { consider(&r.counts); }
    if print_total { consider(total); }

    let has_stdin = results.iter().any(|r| r.display_name.as_deref() == Some("-")) || (results.len() == 1 && opts.files.is_empty());
    if has_stdin && nflags > 1 {
        max_w = max_w.max(7);
    }

    [max_w; 5]
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
                first = false;
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
    let used_files0 = opts.files_from.is_some();

    if let Some(spec) = opts.files_from.clone() {
        if !opts.files.is_empty() {
            eprintln!(
                "wc: extra operand {:?}\nfile operands cannot be combined with --files0-from",
                opts.files[0]
            );
            std::process::exit(1);
        }
        file_list = read_files0_from(&spec)?;
    } else if opts.files.is_empty() {
        file_list = vec![];
    } else {
        file_list = opts.files.clone();
    }

    opts.files = file_list.clone();

    let mut total = Counts::default();
    let mut nfiles_processed = 0u64;
    let mut results: Vec<FileResult> = Vec::new();

    if file_list.is_empty() {
        match count_path(None, &opts) {
            Ok(c) => {
                total = c;
                nfiles_processed += 1;
                results.push(FileResult { counts: c, display_name: None });
            }
            Err(e) => {
                eprintln!("wc: -: {e}");
                ok = false;
            }
        }
    } else {
        for f in &file_list {
            let display = f.to_string_lossy().into_owned();
            match count_path(Some(f), &opts) {
                Ok(c) => {
                    total.lines += c.lines;
                    total.words += c.words;
                    total.chars += c.chars;
                    total.bytes += c.bytes;
                    if c.max_line_length > total.max_line_length { total.max_line_length = c.max_line_length; }
                    nfiles_processed += 1;
                    results.push(FileResult { counts: c, display_name: Some(display) });
                }
                Err(e) => {
                    eprintln!("wc: {display}: {e}");
                    ok = false;
                }
            }
        }
    }

    let widths = compute_widths(&opts, &results, &total, used_files0);

    let print_total = match opts.total_mode {
        TotalMode::Never => false,
        TotalMode::Always | TotalMode::Only => true,
        TotalMode::Auto => nfiles_processed > 1,
    };

    if opts.total_mode != TotalMode::Only {
        for r in &results {
            write_counts(&mut out, &opts, &r.counts, &widths, r.display_name.as_deref())?;
        }
    }

    if print_total {
        let name = if opts.total_mode == TotalMode::Only { None } else { Some("total") };
        write_counts(&mut out, &opts, &total, &widths, name)?;
    }

    out.flush()?;
    Ok(ok)
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

