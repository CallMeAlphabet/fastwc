# fastwc
fastwc — a fast wc rewrite

## Table of Contents
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Uninstall](#Uninstall)
- [Usage](#usage)
- [Testing Conditions](#testing-conditions)

## Quick Start
- **On Arch**
```bash
paru -S fastwc
```
- **Non-arch**
```bash
cargo install fastwc
```
> **Note**: Make sure `~/.cargo/bin` is in your `PATH`. It's added automatically by rustup, but if `fastwc` isn't found, add this to your shell config file:
> ```bash
> # If you use Bash:
> export PATH="$HOME/.cargo/bin:$PATH"
>
> # If you use Fish:
> fish_add_path $HOME/.cargo/bin
>
> # If you use Zsh:
> export PATH="$PATH:$HOME/.cargo/bin"
> ```

```bash
fastwc /path/to/file
```

## Benchmarks

| Input | Mode | Tool | Mean wall time | Opponent relative to fastwc |
|---|---|---|---:|---:|
| **random-5GiB.bin, one file** | **default** | **fastwc 0.2.7** | **1.169 ± 0.171 s** | **—** |
| random-5GiB.bin, one file | default | GNU wc | 84.274 ± 1.053 s | 72.1× slower |
| random-5GiB.bin, one file | default | cw | 15.124 ± 0.137 s | 12.9× slower |
| **random-5GiB.parts, four files** | **default** | **fastwc 0.2.7** | **3.957 ± 0.285 s** | **—** |
| random-5GiB.parts, four files | default | GNU wc | 89.419 ± 4.285 s | 22.6× slower |
| random-5GiB.parts, four files | default | cw | 7.344 ± 0.775 s | 1.9× slower |

Reproduce with Hyperfine:

```bash
hyperfine --shell=default --warmup 0 --runs 2 \
  'LC_ALL=C.utf8 ./target/release/fastwc "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null' \
  'LC_ALL=C.utf8 wc "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null' \
  'LC_ALL=C.utf8 /tmp/cw-0.7.0/bin/cw --threads=4 "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null'

mkdir -p "$HOME/benchmark-inputs/random-5GiB.parts"
split -n 4 -d -a 3 \
  "$HOME/benchmark-inputs/random-5GiB.bin" \
  "$HOME/benchmark-inputs/random-5GiB.parts/part-"

hyperfine --shell=default --warmup 0 --runs 2 \
  'LC_ALL=C.utf8 ./target/release/fastwc "$HOME"/benchmark-inputs/random-5GiB.parts/part-* > /dev/null' \
  'LC_ALL=C.utf8 wc "$HOME"/benchmark-inputs/random-5GiB.parts/part-* > /dev/null' \
  'LC_ALL=C.utf8 /tmp/cw-0.7.0/bin/cw --threads=4 "$HOME"/benchmark-inputs/random-5GiB.parts/part-* > /dev/null'
```

## Uninstall
```bash
cargo uninstall fastwc
```

## Usage
`fastwc` does not add any additional flags to the original `wc`. All flags are the same and behave the same. The --help message was changed though.

## Testing Conditions

https://gist.github.com/CallMeAlphabet/4b7022c4b1a8849e6943526de6a23582
