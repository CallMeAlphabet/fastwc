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

These are Hyperfine wall-clock means from 2 runs with `LC_ALL=C.utf8`, default counting mode, and stdout redirected to `/dev/null`. Times are reported as mean ± standard deviation.

| File | Mode | Tool | Mean wall time | fastwc speedup |
|---|---|---|---:|---:|
| random-5GiB.bin | default | fastwc 0.2.7 | **4.790 ± 0.673 s** | — |
| random-5GiB.bin | default | GNU wc | 81.301 ± 0.209 s | 17.0× |
| random-5GiB.bin | default | cw | 15.711 ± 0.032 s | 3.3× |

Reproduce with Hyperfine:

```bash
hyperfine --shell=default --warmup 0 --runs 2   'LC_ALL=C.utf8 ./target/release/fastwc "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null'   'LC_ALL=C.utf8 wc "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null'   'LC_ALL=C.utf8 /tmp/cw-0.7.0/bin/cw --threads=2 "$HOME/benchmark-inputs/random-5GiB.bin" > /dev/null'
```

## Uninstall
```bash
cargo uninstall fastwc
```

## Usage
`fastwc` does not add any additional flags to the original `wc`. All flags are the same and behave the same. The --help message was changed though.

## Testing Conditions

https://gist.github.com/CallMeAlphabet/4b7022c4b1a8849e6943526de6a23582
