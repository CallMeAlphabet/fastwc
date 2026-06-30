# fastcp
fastcp — a fast wc rewrite

## Table of Contents
- [Quick Start](#quick-start)
- [Uninstall](#Uninstall)
- [Usage](#usage)

## Quick Start
```bash
# Install
cargo install --git https://github.com/CallMeAlphabet/fastwc
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

## Uninstall
```bash
cargo uninstall fastwc
```

## Usage
`fastcp` does not add any additional flags to the original `wc`. All flags are the same and behave the same.
