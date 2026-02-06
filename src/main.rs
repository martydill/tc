use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, p50k_edit, r50k_base};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "tc")]
#[command(about = "Token counter - like wc but for tokens", long_about = None)]
#[command(disable_help_flag = true)]
struct Cli {
    /// Files or directories to count tokens in (reads from stdin if none provided)
    files: Vec<String>,

    /// Also print line count
    #[arg(short = 'l', long)]
    lines: bool,

    /// Also print word count
    #[arg(short = 'w', long)]
    words: bool,

    /// Also print character count
    #[arg(short = 'm', long)]
    chars: bool,

    /// Also print byte count
    #[arg(short = 'c', long)]
    bytes: bool,

    /// Print counts in human-readable format (e.g., 15.2k)
    #[arg(short = 'h', long = "human-readable")]
    human: bool,

    /// Recursively process directories
    #[arg(short = 'r', short_alias = 'R', long)]
    recursive: bool,

    /// Tokenizer encoding to use
    #[arg(short = 'e', long, default_value = "cl100k")]
    encoding: String,

    /// List available encodings
    #[arg(long)]
    list_encodings: bool,

    /// Print help
    #[arg(long, action = clap::ArgAction::Help)]
    help: Option<bool>,
}

#[derive(Default)]
struct Counts {
    tokens: usize,
    lines: usize,
    words: usize,
    chars: usize,
    bytes: usize,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.tokens += other.tokens;
        self.lines += other.lines;
        self.words += other.words;
        self.chars += other.chars;
        self.bytes += other.bytes;
    }
}

fn count_tokens(text: &str, encoding: &str) -> Result<usize> {
    let bpe = match encoding {
        "cl100k" | "cl100k_base" => cl100k_base()?,
        "o200k" | "o200k_base" => o200k_base()?,
        "p50k" | "p50k_base" => p50k_base()?,
        "p50k_edit" => p50k_edit()?,
        "r50k" | "r50k_base" => r50k_base()?,
        _ => anyhow::bail!(
            "Unknown encoding: {}. Use --list-encodings to see available options.",
            encoding
        ),
    };
    Ok(bpe.encode_with_special_tokens(text).len())
}

fn count_all(text: &str, encoding: &str) -> Result<Counts> {
    Ok(Counts {
        tokens: count_tokens(text, encoding)?,
        lines: text.lines().count(),
        words: text.split_whitespace().count(),
        chars: text.chars().count(),
        bytes: text.len(),
    })
}

fn format_human(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_count(n: usize, human: bool) -> String {
    if human {
        format!("{:>7}", format_human(n))
    } else {
        format!("{:>7}", n)
    }
}

fn print_counts(counts: &Counts, name: Option<&str>, cli: &Cli) {
    let mut parts = Vec::new();

    if cli.lines {
        parts.push(format_count(counts.lines, cli.human));
    }
    if cli.words {
        parts.push(format_count(counts.words, cli.human));
    }
    if cli.chars {
        parts.push(format_count(counts.chars, cli.human));
    }
    if cli.bytes {
        parts.push(format_count(counts.bytes, cli.human));
    }

    // Always show tokens (this is our main purpose)
    parts.push(format_count(counts.tokens, cli.human));

    let output = parts.join(" ");

    match name {
        Some(n) => println!("{} {}", output, n),
        None => println!("{}", output),
    }
}

fn list_encodings() {
    println!("Available tokenizer encodings:");
    println!();
    println!("  cl100k, cl100k_base  - GPT-4, GPT-3.5-turbo, text-embedding-ada-002 (default)");
    println!("  o200k, o200k_base    - GPT-4o, GPT-4o-mini");
    println!("  p50k, p50k_base      - Codex models, text-davinci-002, text-davinci-003");
    println!("  p50k_edit            - edit models like text-davinci-edit-001");
    println!("  r50k, r50k_base      - GPT-3 models like davinci");
}

/// Collect all files from a path (file or directory)
fn collect_files(path: &Path, recursive: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        if recursive {
            for entry in WalkDir::new(path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    files.push(entry.path().to_path_buf());
                }
            }
        } else {
            // Non-recursive: only immediate children
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        files.push(entry_path);
                    }
                }
            }
        }
    }

    files.sort();
    files
}

fn process_file(path: &Path, encoding: &str) -> Result<Counts> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    count_all(&content, encoding)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_encodings {
        list_encodings();
        return Ok(());
    }

    let mut total = Counts::default();
    let mut file_count = 0;

    if cli.files.is_empty() {
        // Read from stdin
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        let counts = count_all(&buffer, &cli.encoding)?;
        print_counts(&counts, None, &cli);
    } else {
        // Collect all files from all input paths
        let mut all_files = Vec::new();
        for input in &cli.files {
            let path = Path::new(input);
            if !path.exists() {
                eprintln!("tc: {}: No such file or directory", input);
                continue;
            }
            all_files.extend(collect_files(path, cli.recursive));
        }

        for file_path in &all_files {
            match process_file(file_path, &cli.encoding) {
                Ok(counts) => {
                    total.add(&counts);
                    file_count += 1;
                    print_counts(&counts, Some(&file_path.display().to_string()), &cli);
                }
                Err(e) => {
                    // Skip binary/unreadable files with a warning
                    eprintln!("tc: {}: {}", file_path.display(), e);
                }
            }
        }

        if file_count > 1 {
            print_counts(&total, Some("total"), &cli);
        }
    }

    Ok(())
}
