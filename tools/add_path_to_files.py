#!/usr/bin/env python3
import argparse
import re
import sys
import tokenize
from pathlib import Path
from typing import Tuple

ENCODING_RE = re.compile(rb"^[ \t\f]*#.*coding[:=][ \t]*([-\w.]+)")

def detect_encoding(path: Path) -> str:
    with path.open("rb") as fb:
        enc, _ = tokenize.detect_encoding(fb.readline)
    return enc

def read_text(path: Path, encoding: str) -> Tuple[str, str]:
    """Return (text, newline_style)."""
    data = path.read_bytes()
    newline = "\r\n" if b"\r\n" in data and b"\n" in data else "\n"
    return data.decode(encoding), newline

def has_encoding_line(lines: list[str]) -> int:
    """
    Return index (0 or 1) of an encoding declaration if present in the first two lines,
    else -1. Uses PEP 263 patterns.
    """
    for i in range(min(2, len(lines))):
        if ENCODING_RE.match(lines[i].encode("utf-8", "surrogatepass")):
            return i
    return -1

def compute_insertion_index(lines: list[str]) -> int:
    """
    Insert at top, but:
      - if first line is shebang ('#!'), insert after it;
      - if an encoding cookie exists in the first two lines, insert AFTER it
        so the cookie remains on line 1 or 2 as required by PEP 263.
    """
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    enc_idx = has_encoding_line(lines)
    if enc_idx != -1 and idx <= enc_idx:
        idx = enc_idx + 1
    return idx

def already_has_comment(lines: list[str], comment_line: str) -> bool:
    for i in range(min(3, len(lines))):
        if lines[i].rstrip("\r\n") == comment_line:
            return True
    return False

def process_file(path: Path, root: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Returns True if file was modified.
    """
    rel = path.relative_to(root).as_posix()
    comment_line = f"# {rel}"

    encoding = detect_encoding(path)
    text, newline = read_text(path, encoding)
    lines = text.splitlines()

    insert_at = compute_insertion_index(lines)

    if lines and lines[0].startswith("#!") and verbose:
        print(f"[shebang] {rel}", file=sys.stderr)

    if already_has_comment(lines, comment_line):
        if verbose:
            print(f"[skip] already has comment: {rel}", file=sys.stderr)
        return False

    new_lines = lines[:insert_at] + [comment_line] + lines[insert_at:]

    output = (newline.join(new_lines) + newline) if new_lines else comment_line + "\n"

    if dry_run:
        print(f"[dry-run] would modify: {rel}", file=sys.stderr)
        return False

    path.write_text(output, encoding=encoding, newline=newline)
    if verbose:
        print(f"[write] {rel}", file=sys.stderr)
    return True

def should_skip_dir(p: Path) -> bool:
    name = p.name
    return name in {
        ".git", "__pycache__", ".venv", "venv", "env", "build", "dist", ".mypy_cache",
        ".pytest_cache"
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Insert relative path comment at top of Python files.")
    ap.add_argument(
        "root", nargs="?", default=".", help="Root directory to scan (default: current directory)."
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Scan and report changes without writing files."
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose progress to stderr.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: root '{root}' is not a directory", file=sys.stderr)
        sys.exit(2)

    modified = 0
    total = 0

    for p in root.rglob("*.py"):
        if any(should_skip_dir(parent) for parent in p.parents):
            continue
        total += 1
        if process_file(p, root, dry_run=args.dry_run, verbose=args.verbose):
            modified += 1

    print(f"Processed {total} Python files; modified {modified}.", file=sys.stderr)

if __name__ == "__main__":
    main()
