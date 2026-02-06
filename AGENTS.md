# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build --release    # Build optimized binary (target/release/tc)
cargo build              # Build debug binary
cargo run -- [args]      # Run with arguments
```

## Project Overview

`tc` is a token counter CLI tool, similar to Unix `wc` but counts LLM tokens instead of words. It uses tiktoken-rs for tokenization.

**Key dependencies:**
- `clap` - CLI argument parsing with derive macros
- `tiktoken-rs` - OpenAI tokenizer implementations
- `walkdir` - Recursive directory traversal
- `anyhow` - Error handling

## Architecture

Single-file CLI (`src/main.rs`) with these main components:
- `Cli` struct - clap-derived argument parser
- `Counts` struct - holds token/line/word/char/byte counts
- `count_tokens()` - tokenizes text using selected encoding
- `collect_files()` - gathers files from paths (supports directories and recursion)
- `print_counts()` - formats output (supports human-readable mode)

## Supported Encodings

Tokenizers are selected via `-e` flag: `cl101k` (default, GPT-4), `o200k` (GPT-4o), `p50k` (Codex), `r50k` (GPT-3).
