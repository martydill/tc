# tc - wc for tokens!

A CLI tool for counting tokens in files or directories.

## Installation

```bash
cargo build --release
# Binary at target/release/tc
```

## Usage

```bash
tc [OPTIONS] [FILES]...
```

**Examples:**
```bash
echo "Hello world" | tc          # Count tokens from stdin
tc file.txt                      # Count tokens in a file
tc file1.txt file2.txt           # Multiple files (shows total)
tc src/                          # All files in a directory
tc -r .                          # Recursive directory search
tc -lwc file.txt                 # Include lines, words, bytes
tc -h file.txt                   # Human-readable output (1.5k)
tc -e o200k file.txt             # Use GPT-4o tokenizer
```

## Options

| Flag | Description |
|------|-------------|
| `-l, --lines` | Print line count |
| `-w, --words` | Print word count |
| `-m, --chars` | Print character count |
| `-c, --bytes` | Print byte count |
| `-h, --human-readable` | Human-readable counts (e.g., 15.2k) |
| `-r, -R, --recursive` | Recursively process directories |
| `-e, --encoding` | Tokenizer encoding (default: cl100k) |
| `--list-encodings` | Show available encodings |

## Encodings

- `cl100k` - GPT-4, GPT-3.5-turbo (default)
- `o200k` - GPT-4o, GPT-4o-mini
- `p50k` - Codex models
- `r50k` - GPT-3 models

Run `tc --list-encodings` for full details.

## License

This project is licensed under the MIT License.