import os
import time
import tarfile
from pathlib import Path
from glob import glob

try:
    import pathspec
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pathspec") from exc

GITIGNORE = ".gitignore"
OUTPUT_BASE = "archive"
SCRIPT_NAME = os.path.basename(__file__)
ARCHIVE_ROOT = os.path.basename(os.path.abspath(os.path.dirname(__file__)))

def build_spec(extra_patterns=None):
    """Return a PathSpec built from .gitignore plus any extra patterns."""
    patterns = []
    if Path(GITIGNORE).exists():
        with open(GITIGNORE, "r", encoding="utf-8", errors="ignore") as fh:
            patterns.extend(line.rstrip("\n") for line in fh)

    if extra_patterns:
        patterns.extend(extra_patterns)

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

def should_ignore(spec, rel_path, is_dir=False):
    """
    Decide whether rel_path should be ignored.
    Directories must end with '/' for gitwildmatch directory patterns.
    """
    posix_path = rel_path.as_posix()
    if is_dir:
        posix_path += "/"
    return spec.match_file(posix_path)

def create_archive(output_name, include_patterns=None):
    """Walk tree, prune ignored dirs, and write archive. Then explicitly add include patterns."""
    include_patterns = include_patterns or []
    extra = [
        f"{OUTPUT_BASE}_*.tar.gz",
        ".git",
        ".git/**",
        f"**/{SCRIPT_NAME}",
    ]
    spec = build_spec(extra)

    print("Active ignore patterns:")
    for p in spec.patterns:
        print(f"- {p.pattern}")
    if include_patterns:
        print("\nExplicit include patterns (will be forced into archive):")
        for ip in include_patterns:
            print(f"- {ip}")
    print()

    added = set()

    with tarfile.open(output_name, "w:gz") as tar:
        for root, dirs, files in os.walk(".", topdown=True):
            root_path = Path(root)

            dirs[:] = [
                d
                for d in dirs
                if not should_ignore(spec, root_path.joinpath(d).relative_to("."), True)
            ]

            for name in files:
                file_path = root_path / name
                rel = file_path.relative_to(".")
                rel_posix = rel.as_posix()

                if rel.name == output_name:
                    continue

                if should_ignore(spec, rel, False):
                    print(f" -> Excluding {rel}")
                    continue

                tar.add(file_path, arcname=str(Path(ARCHIVE_ROOT) / rel))
                added.add(rel_posix)
                print(f" -> Including {rel}")

        for patt in include_patterns:
            matches = glob(patt, recursive=True)
            if not matches:
                print(f" -> Include pattern matched nothing: {patt}")
                continue

            for m in matches:
                p = Path(m)
                if p.is_dir():
                    for dir_root, _dirs, dir_files in os.walk(p, topdown=True):
                        for fname in dir_files:
                            fpath = Path(dir_root) / fname
                            rel = fpath.relative_to(".")
                            rel_posix = rel.as_posix()
                            if rel.name == output_name:
                                continue
                            if rel_posix in added:
                                continue
                            tar.add(fpath, arcname=str(Path(ARCHIVE_ROOT) / rel))
                            added.add(rel_posix)
                            print(f" -> Forcing include (dir): {rel}")
                else:
                    if not p.exists():
                        print(f" -> Include target not found: {p}")
                        continue
                    rel = p.relative_to(".")
                    rel_posix = rel.as_posix()
                    if rel.name == output_name:
                        continue
                    if rel_posix in added:
                        continue
                    tar.add(p, arcname=str(Path(ARCHIVE_ROOT) / rel))
                    added.add(rel_posix)
                    print(f" -> Forcing include: {rel}")

    size_kib = os.path.getsize(output_name) / 1024
    print(f"\nCreated {output_name} ({size_kib:.2f} KiB).")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        help="pattern to force-include; can be passed multiple times",
    )
    args, _ = parser.parse_known_args()

    out_name = f"{OUTPUT_BASE}.tar.gz"
    try:
        create_archive(out_name, include_patterns=args.include)
    except Exception as err:
        print(f"Error: {err}")
