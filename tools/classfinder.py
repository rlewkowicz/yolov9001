#!/usr/bin/env python3
"""
Compare Python entities (classes, functions, methods) between a TARGET folder and one or more COMPARE folders.

Usage:
  python compare_entities.py --target path/to/target \
      --compare libA --compare libB,libC

What it does:
  1) Crawls TARGET and builds a map of all classes, functions, and methods per file.
  2) Crawls all COMPARE folders, builds the same map, and **distills to unique entities**
     (we don’t care which compare file they came from; duplicates are merged).
  3) Matches TARGET entities by name (and class for methods) against the **union** of unique entities from COMPARE.
  4) Prints a final report **organized by TARGET file** to STDOUT:
       - the target entity’s code
       - its unique counterparts from the compare folders (deduped by content)

Notes:
  - All progress logs and non-final output go to STDERR.
  - Gracefully skips files that fail to parse.
"""

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Set

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def read_text(p: str) -> str:
    return Path(p).read_text(encoding="utf-8", errors="replace")

def iter_python_files(folder_path: str) -> Iterable[str]:
    root = Path(folder_path)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".py"):
                yield str(Path(dp) / fn)

@dataclass(frozen=True)
class EntityKey:
    """
    Unique identity for an entity by kind and name.
    - kind: 'class' | 'function' | 'method'
    - name: for class => class name
            for function => function name
            for method => method name (paired with cls for identity)
    - cls:  class name for methods; otherwise None
    """
    kind: str
    name: str
    cls: Optional[str] = None

    def pretty(self) -> str:
        if self.kind == "class":
            return f"class {self.name}"
        if self.kind == "function":
            return f"def {self.name}(...)"  # signature not stored; name only
        if self.kind == "method":
            return f"class {self.cls} :: def {self.name}(...)"
        return f"{self.kind} {self.name}"

@dataclass
class EntityInstance:
    """
    A concrete occurrence of an entity in a file, carrying its source code block.
    """
    key: EntityKey
    file_path: str
    code: str

def _node_block(n: ast.AST, lines: List[str]) -> str:
    start = getattr(n, "lineno", 1)
    if hasattr(n, "decorator_list") and getattr(n, "decorator_list"):
        start = min([d.lineno for d in n.decorator_list] + [start])
    end = getattr(n, "end_lineno", None)
    if end is None:
        end = start
        if hasattr(n, "body") and isinstance(n.body, list) and n.body:
            end = getattr(n.body[-1], "end_lineno", getattr(n.body[-1], "lineno", end))
    start = max(1, int(start))
    end = max(start, int(end))
    return "\n".join(lines[start - 1:end])

def extract_entities_from_file(file_path: str) -> List[EntityInstance]:
    """
    Extract classes, functions (module-level), and methods (inside classes)
    from a single Python file.
    """
    try:
        src = read_text(file_path)
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []  # skip files that don't parse
    lines = src.splitlines()

    out: List[EntityInstance] = []

    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            key = EntityKey(kind="class", name=n.name)
            out.append(EntityInstance(key=key, file_path=file_path, code=_node_block(n, lines)))
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mkey = EntityKey(kind="method", name=m.name, cls=n.name)
                    out.append(
                        EntityInstance(key=mkey, file_path=file_path, code=_node_block(m, lines))
                    )
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = EntityKey(kind="function", name=n.name)
            out.append(EntityInstance(key=key, file_path=file_path, code=_node_block(n, lines)))

    return out

def normalize_code(code: str) -> str:
    """
    Normalize code for deduping: trim ends and strip trailing spaces per line.
    """
    lines = [ln.rstrip() for ln in code.strip().splitlines()]
    return "\n".join(lines).strip()

def build_folder_index(folder: str) -> Dict[str, List[EntityInstance]]:
    """
    Return: map[target_file_path] -> list of EntityInstance found in that file.
    """
    index: Dict[str, List[EntityInstance]] = {}
    for f in iter_python_files(folder):
        ents = extract_entities_from_file(f)
        if not ents:
            continue
        index.setdefault(f, []).extend(ents)
    return index

def distill_unique_entities_from_indexes(
    indexes: List[Dict[str, List[EntityInstance]]]
) -> Dict[EntityKey, Dict[str, str]]:
    """
    Given several folder indexes (typically the COMPARE folders),
    return a mapping:
        EntityKey -> { normalized_code : original_example_code }
    i.e., the union of unique entities across all compare folders,
    deduped by code content (we don't care which file they came from).
    """
    distilled: Dict[EntityKey, Dict[str, str]] = {}
    for idx in indexes:
        for file_path, ents in idx.items():
            for e in ents:
                norm = normalize_code(e.code)
                bucket = distilled.setdefault(e.key, {})
                bucket.setdefault(norm, e.code)  # store first seen original
    return distilled

def print_final_report(
    target_index: Dict[str, List[EntityInstance]],
    compare_distilled: Dict[EntityKey, Dict[str, str]],
) -> None:
    """
    Print final report organized by TARGET file to STDOUT.
    If there are no matches, prints nothing (logs to STDERR instead).
    """
    any_output = False
    for file_path in sorted(target_index.keys()):
        file_entities = target_index[file_path]
        matches = [(e, compare_distilled.get(e.key, {})) for e in file_entities]
        matches = [(e, uniq_map) for (e, uniq_map) in matches if uniq_map]

        if not matches:
            continue

        any_output = True
        print(f"\n=== TARGET FILE: {file_path}")
        for e, uniq_map in matches:
            print(f"\n--- ENTITY: {e.key.pretty()}")
            print(e.code)
            codes = list(uniq_map.values())
            print(f"\n>>> UNIQUE COUNTERPARTS FROM COMPARE ({len(codes)})")
            for i, c in enumerate(codes, 1):
                print(f"\n-- variant {i} --")
                print(c)

    if not any_output:
        log("[info] No overlapping entities (by name/type) found between TARGET and COMPARE.")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare entities (classes, functions, methods) in a target folder "
        "against one or more compare folders. Distills compare entities to unique sets."
    )
    ap.add_argument(
        "--target", required=True, help="Path to the TARGET folder (will be crawled recursively)."
    )
    ap.add_argument(
        "--compare",
        action="append",
        required=True,
        metavar="FOLDER[,FOLDER2,...]",
        help="Repeatable flag or comma-separated list of COMPARE folders."
    )
    return ap.parse_args()

def flatten_compare_args(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",")]
        out.extend([p for p in parts if p])
    seen: Set[str] = set()
    unique: List[str] = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique

def main() -> None:
    args = parse_args()
    target_folder = str(Path(args.target).resolve())
    compare_folders = [str(Path(p).resolve()) for p in flatten_compare_args(args.compare)]

    log(">> Step 1/4: Validating paths...")
    if not Path(target_folder).is_dir():
        log(f"[error] Target folder does not exist or is not a directory: {target_folder}")
        raise SystemExit(2)
    missing = [p for p in compare_folders if not Path(p).is_dir()]
    if missing:
        log(f"[error] Compare folder(s) not found: {', '.join(missing)}")
        raise SystemExit(2)
    log(f"[ok] target={target_folder}")
    log(f"[ok] compare={', '.join(compare_folders)}")

    log(">> Step 2/4: Scanning TARGET for Python entities...")
    target_index = build_folder_index(target_folder)
    target_file_count = len(target_index)
    target_entity_count = sum(len(v) for v in target_index.values())
    log(
        f"[ok] TARGET files with entities: {target_file_count}, total entities: {target_entity_count}"
    )

    log(">> Step 3/4: Scanning COMPARE folders for Python entities...")
    compare_indexes = []
    for cf in compare_folders:
        log(f"   - scanning: {cf}")
        idx = build_folder_index(cf)
        files_with = len(idx)
        ents = sum(len(v) for v in idx.values())
        log(f"     [ok] files with entities: {files_with}, total entities: {ents}")
        compare_indexes.append(idx)

    log(">> Step 4/4: Distilling COMPARE entities into unique sets...")
    compare_distilled = distill_unique_entities_from_indexes(compare_indexes)
    classes = sum(1 for k in compare_distilled.keys() if k.kind == "class")
    functions = sum(1 for k in compare_distilled.keys() if k.kind == "function")
    methods = sum(1 for k in compare_distilled.keys() if k.kind == "method")
    log(
        f"[ok] Distilled unique entities (by name/type): classes={classes}, functions={functions}, methods={methods}"
    )

    log(">> Generating final report (organized by TARGET file)...")
    print_final_report(target_index, compare_distilled)
    log(">> Done.")

if __name__ == "__main__":
    main()
