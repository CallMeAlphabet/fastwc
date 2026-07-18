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
# or fastwc-bin for a prebuilt release
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
# Use it!
fastwc /path/to/file
```

## Benchmarks

### Benchmark 1: 1.5 GiB file

| Tool | Time | Speed vs fastwc |
|------|------|------------------|
| **fastwc** | **0.16s** | **1x (baseline)** |
| wc | 23s | 141.6x slower |

***The gains in speed are higher the bigger the file.***

## Uninstall
```bash
cargo uninstall fastwc
```

## Usage
`fastwc` does not add any additional flags to the original `wc`. All flags are the same and behave the same. The --help message was changed though.

## Testing Conditions

https://gist.github.com/CallMeAlphabet/4b7022c4b1a8849e6943526de6a23582
