use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use tiktoken_rs::CoreBPE;
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

    /// Follow symbolic links when recursing directories
    #[arg(short = 'L', long)]
    follow_links: bool,

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

fn create_bpe(encoding: &str) -> Result<CoreBPE> {
    match encoding {
        "cl100k" | "cl100k_base" => Ok(cl100k_base()?),
        "o200k" | "o200k_base" => Ok(o200k_base()?),
        "p50k" | "p50k_base" => Ok(p50k_base()?),
        "p50k_edit" => Ok(p50k_edit()?),
        "r50k" | "r50k_base" => Ok(r50k_base()?),
        _ => anyhow::bail!(
            "Unknown encoding: {}. Use --list-encodings to see available options.",
            encoding
        ),
    }
}

fn count_all(text: &str, bpe: &CoreBPE) -> Counts {
    Counts {
        tokens: bpe.encode_with_special_tokens(text).len(),
        lines: text.lines().count(),
        words: text.split_whitespace().count(),
        chars: text.chars().count(),
        bytes: text.len(),
    }
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
fn collect_files(path: &Path, recursive: bool, follow_links: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        if recursive {
            for entry in WalkDir::new(path)
                .follow_links(follow_links)
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

fn process_file(path: &Path, bpe: &CoreBPE) -> Result<Counts> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    Ok(count_all(&content, bpe))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    // --- Counts ---

    #[test]
    fn counts_default_is_zero() {
        let c = Counts::default();
        assert_eq!(c.tokens, 0);
        assert_eq!(c.lines, 0);
        assert_eq!(c.words, 0);
        assert_eq!(c.chars, 0);
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn counts_add_accumulates() {
        let mut a = Counts {
            tokens: 1,
            lines: 2,
            words: 3,
            chars: 4,
            bytes: 5,
        };
        let b = Counts {
            tokens: 10,
            lines: 20,
            words: 30,
            chars: 40,
            bytes: 50,
        };
        a.add(&b);
        assert_eq!(a.tokens, 11);
        assert_eq!(a.lines, 22);
        assert_eq!(a.words, 33);
        assert_eq!(a.chars, 44);
        assert_eq!(a.bytes, 55);
    }

    #[test]
    fn counts_add_zero_is_identity() {
        let mut a = Counts {
            tokens: 5,
            lines: 10,
            words: 15,
            chars: 20,
            bytes: 25,
        };
        let zero = Counts::default();
        a.add(&zero);
        assert_eq!(a.tokens, 5);
        assert_eq!(a.lines, 10);
        assert_eq!(a.words, 15);
        assert_eq!(a.chars, 20);
        assert_eq!(a.bytes, 25);
    }

    // --- create_bpe ---

    #[test]
    fn create_bpe_valid_encodings() {
        for name in &[
            "cl100k",
            "cl100k_base",
            "o200k",
            "o200k_base",
            "p50k",
            "p50k_base",
            "p50k_edit",
            "r50k",
            "r50k_base",
        ] {
            assert!(
                create_bpe(name).is_ok(),
                "Expected valid encoding for '{}'",
                name
            );
        }
    }

    #[test]
    fn create_bpe_invalid_encoding() {
        let result = create_bpe("nonexistent");
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error for unknown encoding"),
        };
        assert!(msg.contains("Unknown encoding"));
        assert!(msg.contains("nonexistent"));
    }

    // --- count_all ---

    #[test]
    fn count_all_empty_string() {
        let bpe = cl100k_base().unwrap();
        let c = count_all("", &bpe);
        assert_eq!(c.tokens, 0);
        assert_eq!(c.lines, 0);
        assert_eq!(c.words, 0);
        assert_eq!(c.chars, 0);
        assert_eq!(c.bytes, 0);
    }

    #[test]
    fn count_all_single_word() {
        let bpe = cl100k_base().unwrap();
        let c = count_all("hello", &bpe);
        assert!(c.tokens >= 1);
        assert_eq!(c.lines, 1);
        assert_eq!(c.words, 1);
        assert_eq!(c.chars, 5);
        assert_eq!(c.bytes, 5);
    }

    #[test]
    fn count_all_multiple_lines() {
        let bpe = cl100k_base().unwrap();
        let c = count_all("line one\nline two\nline three", &bpe);
        assert_eq!(c.lines, 3);
        assert_eq!(c.words, 6);
    }

    #[test]
    fn count_all_unicode() {
        let bpe = cl100k_base().unwrap();
        let text = "café résumé";
        let c = count_all(text, &bpe);
        assert_eq!(c.words, 2);
        // chars != bytes for multi-byte UTF-8
        assert_eq!(c.chars, 11); // c-a-f-é- -r-é-s-u-m-é
        assert!(c.bytes > c.chars); // é is 2 bytes in UTF-8
        assert!(c.tokens >= 1);
    }

    #[test]
    fn count_all_whitespace_only() {
        let bpe = cl100k_base().unwrap();
        let c = count_all("   \n  \n  ", &bpe);
        assert_eq!(c.words, 0);
        assert_eq!(c.lines, 3);
    }

    #[test]
    fn count_all_different_encodings_may_differ() {
        let cl = cl100k_base().unwrap();
        let o2 = o200k_base().unwrap();
        let text = "The quick brown fox jumps over the lazy dog";
        let c1 = count_all(text, &cl);
        let c2 = count_all(text, &o2);
        // Non-token counts must be identical
        assert_eq!(c1.lines, c2.lines);
        assert_eq!(c1.words, c2.words);
        assert_eq!(c1.chars, c2.chars);
        assert_eq!(c1.bytes, c2.bytes);
        // Token counts may differ between encodings
        assert!(c1.tokens >= 1);
        assert!(c2.tokens >= 1);
    }

    // --- format_human ---

    #[test]
    fn format_human_small_numbers() {
        assert_eq!(format_human(0), "0");
        assert_eq!(format_human(1), "1");
        assert_eq!(format_human(999), "999");
    }

    #[test]
    fn format_human_thousands() {
        assert_eq!(format_human(1_000), "1.0k");
        assert_eq!(format_human(1_500), "1.5k");
        assert_eq!(format_human(15_200), "15.2k");
        assert_eq!(format_human(999_999), "1000.0k");
    }

    #[test]
    fn format_human_millions() {
        assert_eq!(format_human(1_000_000), "1.0M");
        assert_eq!(format_human(2_500_000), "2.5M");
        assert_eq!(format_human(999_999_999), "1000.0M");
    }

    #[test]
    fn format_human_billions() {
        assert_eq!(format_human(1_000_000_000), "1.0G");
        assert_eq!(format_human(3_700_000_000), "3.7G");
    }

    // --- format_count ---

    #[test]
    fn format_count_plain() {
        assert_eq!(format_count(42, false), "     42");
    }

    #[test]
    fn format_count_human_mode() {
        assert_eq!(format_count(42, true), "     42");
        assert_eq!(format_count(1_500, true), "   1.5k");
    }

    // --- collect_files ---

    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "tc_test_{}_{}_{}",
            std::process::id(),
            id,
            std::thread::current().name().unwrap_or("unknown")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn collect_files_single_file() {
        let dir = create_temp_dir();
        let file = dir.join("test.txt");
        fs::write(&file, "hello").unwrap();

        let files = collect_files(&file, false, false);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn collect_files_directory_non_recursive() {
        let dir = create_temp_dir();
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.txt"), "b").unwrap();
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("c.txt"), "c").unwrap();

        let files = collect_files(&dir, false, false);
        // Should only get a.txt and b.txt, not sub/c.txt
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.parent().unwrap() == dir));

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn collect_files_directory_recursive() {
        let dir = create_temp_dir();
        fs::write(dir.join("a.txt"), "a").unwrap();
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("b.txt"), "b").unwrap();
        let deep = sub.join("deep");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("c.txt"), "c").unwrap();

        let files = collect_files(&dir, true, false);
        assert_eq!(files.len(), 3);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn collect_files_nonexistent_returns_empty() {
        let files = collect_files(Path::new("/nonexistent/path/xyz"), false, false);
        assert!(files.is_empty());
    }

    #[test]
    fn collect_files_results_are_sorted() {
        let dir = create_temp_dir();
        fs::write(dir.join("c.txt"), "c").unwrap();
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.txt"), "b").unwrap();

        let files = collect_files(&dir, false, false);
        let names: Vec<_> = files
            .iter()
            .map(|f| f.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);

        fs::remove_dir_all(&dir).unwrap();
    }

    // --- process_file ---

    #[test]
    fn process_file_valid() {
        let dir = create_temp_dir();
        let file = dir.join("hello.txt");
        fs::write(&file, "hello world").unwrap();

        let bpe = cl100k_base().unwrap();
        let counts = process_file(&file, &bpe).unwrap();
        assert!(counts.tokens >= 1);
        assert_eq!(counts.words, 2);
        assert_eq!(counts.bytes, 11);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn process_file_nonexistent() {
        let bpe = cl100k_base().unwrap();
        let result = process_file(Path::new("/nonexistent/file.txt"), &bpe);
        assert!(result.is_err());
    }

    #[test]
    fn process_file_empty() {
        let dir = create_temp_dir();
        let file = dir.join("empty.txt");
        fs::write(&file, "").unwrap();

        let bpe = cl100k_base().unwrap();
        let counts = process_file(&file, &bpe).unwrap();
        assert_eq!(counts.tokens, 0);
        assert_eq!(counts.words, 0);
        assert_eq!(counts.lines, 0);

        fs::remove_dir_all(&dir).unwrap();
    }

    // --- print_counts (output verification) ---

    #[test]
    fn print_counts_tokens_only() {
        // When no extra flags are set, only tokens column should appear
        let cli = Cli {
            files: vec![],
            lines: false,
            words: false,
            chars: false,
            bytes: false,
            human: false,
            recursive: false,
            follow_links: false,
            encoding: "cl100k".to_string(),
            list_encodings: false,
            help: None,
        };
        let counts = Counts {
            tokens: 42,
            lines: 10,
            words: 20,
            chars: 30,
            bytes: 40,
        };
        // Just verify it doesn't panic with/without a name
        print_counts(&counts, None, &cli);
        print_counts(&counts, Some("test.txt"), &cli);
    }

    #[test]
    fn print_counts_all_flags() {
        let cli = Cli {
            files: vec![],
            lines: true,
            words: true,
            chars: true,
            bytes: true,
            human: true,
            recursive: false,
            follow_links: false,
            encoding: "cl100k".to_string(),
            list_encodings: false,
            help: None,
        };
        let counts = Counts {
            tokens: 1500,
            lines: 100,
            words: 200,
            chars: 3000,
            bytes: 3500,
        };
        // Verify it doesn't panic when all flags are set
        print_counts(&counts, Some("big.txt"), &cli);
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_encodings {
        list_encodings();
        return Ok(());
    }

    if cli.files.is_empty() {
        // Read from stdin
        let bpe = create_bpe(&cli.encoding)?;
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        let counts = count_all(&buffer, &bpe);
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
            all_files.extend(collect_files(path, cli.recursive, cli.follow_links));
        }

        // Initialize tokenizer once, then process files in parallel
        let bpe = create_bpe(&cli.encoding)?;

        let results: Vec<_> = all_files
            .par_iter()
            .map(|file_path| {
                let result = process_file(file_path, &bpe);
                (file_path, result)
            })
            .collect();

        let mut total = Counts::default();
        let mut file_count = 0;

        for (file_path, result) in &results {
            match result {
                Ok(counts) => {
                    total.add(counts);
                    file_count += 1;
                    print_counts(counts, Some(&file_path.display().to_string()), &cli);
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
