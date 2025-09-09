#!/usr/bin/env python3
import os
import subprocess
import pathlib
import argparse
import tempfile
import time
from typing import List, Dict, Tuple, Optional, Set
from openai import OpenAI
from openai import BadRequestError
import re
import shutil  # added

client = OpenAI()

parser = argparse.ArgumentParser(
    description=
    "Upload repo files to an OpenAI Vector Store and request a code review via File Search."
)
parser.add_argument(
    "--diff",
    action="store_true",
    help="Upload only UNSTAGED changes (working tree). Default is all tracked files."
)
parser.add_argument(
    "--prompt", type=str, default="", help="Complete override of the review prompt."
)
parser.add_argument(
    "--add",
    type=str,
    default="",
    help="Additional instructions appended to the default review prompt."
)
parser.add_argument(
    "--structure",
    choices=["repo", "selection", "none"],
    default="selection",
    help="Attach a REPO_STRUCTURE.md manifest: 'repo' = whole repo tree (default), "
    "'selection' = only uploaded files' tree, 'none' = don't attach."
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-5",
    help="Model to use (default: gpt-5). Example alternatives: gpt-4.1, gpt-4o."
)
parser.add_argument(
    "--api-key", type=str, default=None, help="Override OPENAI_API_KEY for this run."
)
parser.add_argument(
    "--root",
    action="append",
    default=None,
    metavar="DIRS",
    help="Override allowed top-level root directories. "
    "Repeat this flag or provide a comma-separated list. "
    "Defaults to: core, models, utils"
)
parser.add_argument(
    "--single",
    action="append",
    default=None,
    metavar="FILES",
    help="Override allowed single files (repo-relative paths). "
    "Repeat this flag or provide a comma-separated list. "
    "Default: 9001.py"
)

args = parser.parse_args()

if args.api_key:
    client = OpenAI(api_key=args.api_key)

repo_root = pathlib.Path.cwd()

ACTION_PLAN_DIRNAME = "action_plan"

def _parse_action_items(text: str) -> List[str]:
    """
    Extract action items from the model output.
    - An item starts on a line that begins with '<number>)'
    - It ends at a line that is exactly '©©©' (optionally with surrounding whitespace)
    The delimiter line is NOT included in the saved item.
    Returns items in encounter order (filenames will be 1.txt, 2.txt, ...).
    """
    items: List[str] = []
    lines = text.splitlines()
    buf: List[str] = []
    in_item = False

    start_re = re.compile(r'^\s*\d+\)\s')  # e.g., "1) Do X"
    end_re = re.compile(r'^\s*©©©\s*$')  # delimiter line

    for line in lines:
        if not in_item:
            if start_re.match(line):
                buf = [line]
                in_item = True
        else:
            if end_re.match(line):
                items.append("\n".join(buf).rstrip())
                buf = []
                in_item = False
            else:
                buf.append(line)

    return items

def _write_action_plan(items: List[str], root: pathlib.Path) -> int:
    """
    Wipes (or creates) the action_plan folder and writes each item as N.txt.
    Returns the number of files written.
    """
    out_dir = root / ACTION_PLAN_DIRNAME
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(items, start=1):
        (out_dir / f"{i}.txt").write_text(item.strip() + "\n", encoding="utf-8")
    return len(items)

def run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, cwd=repo_root).strip()

def list_tracked() -> List[str]:
    out = run(["git", "ls-files"])
    return [p for p in out.splitlines() if p.strip()]

def list_unstaged_only() -> List[str]:
    modified = run(["git", "diff", "--name-only"]).splitlines()
    return sorted(set([p for p in modified if p.strip()]))

def git_status_map() -> Dict[str, str]:
    """
    Map path -> two-letter porcelain code (e.g., ' M', '??', 'A ', 'M ').
    """
    out = run(["git", "status", "--porcelain"])
    status = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:]
        status[path] = code
    return status

candidates = list_unstaged_only() if args.diff else list_tracked()

ALLOWED_EXTS = {
    "c", "cpp", "css", "csv", "doc", "docx", "gif", "go", "html", "java", "jpeg", "jpg", "js",
    "json", "md", "pdf", "php", "pkl", "png", "pptx", "py", "rb", "tar", "tex", "ts", "txt", "webp",
    "xlsx", "xml", "zip"
}

_DEFAULT_ALLOWED_ROOTS = {"core", "models", "utils"}
_DEFAULT_ALLOWED_SINGLE_FILES = {"9001.py", "GEMINI.md"}

def _collect_overrides(values: Optional[List[str]],
                       *,
                       empty_clears: bool = True) -> Optional[Set[str]]:
    """
    values is what argparse gives for action='append', e.g. ['a,b', 'c'] or None.
    - Split on commas and whitespace.
    - Strip tokens, drop empties.
    - If user passes an explicit empty string (e.g. --root ""), return empty set
      so defaults are cleared.
    Returns:
      None  -> flag not provided; use defaults
      set() -> explicit clear
      set{...} -> user-provided override
    """
    if values is None:
        return None

    if empty_clears and all((v or "").strip() == "" for v in values):
        return set()

    tokens: List[str] = []
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        tokens.extend(t for t in re.split(r"[,\s]+", v) if t)

    return set(tokens) if tokens else (set() if empty_clears else None)

_root_override = _collect_overrides(args.root)  # None | set()
_single_override = _collect_overrides(args.single)  # None | set()

ALLOWED_ROOTS = (_root_override if _root_override is not None else set(_DEFAULT_ALLOWED_ROOTS))
ALLOWED_SINGLE_FILES = (
    _single_override if _single_override is not None else set(_DEFAULT_ALLOWED_SINGLE_FILES)
)

def allowed_scope(p: str) -> bool:
    parts = pathlib.PurePosixPath(p).parts
    if not parts:
        return False
    top = parts[0]
    if p in ALLOWED_SINGLE_FILES:
        return True
    return top in ALLOWED_ROOTS

def keep(p: str) -> bool:
    path = (repo_root / p)
    if not path.exists() or not path.is_file():
        return False
    ext = path.suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTS:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:  # <-- skip zero-byte files (prevents OpenAI 400 "File is empty.")
        return False
    return size < 2_000_000  # 2MB cap for review inputs

files = [f for f in candidates if allowed_scope(f) and keep(f)]
empty_skipped = []
for p in candidates:
    path = repo_root / p
    if allowed_scope(p) and path.exists() and path.is_file():
        try:
            if path.stat().st_size == 0:
                empty_skipped.append(p)
        except OSError:
            pass
if empty_skipped:
    print("[info] Skipped zero-byte files:", ", ".join(sorted(empty_skipped)))

if not files:
    raise SystemExit("No matching files to upload (after filtering).")

def build_tree(paths: List[str]) -> str:
    tree = {}
    for p in sorted(paths):
        parts = p.split("/")
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node.setdefault("__files__", []).append(part)
            else:
                node = node.setdefault(part, {})
    lines = []

    def walk(node, prefix=""):
        dirs = sorted([k for k in node.keys() if k != "__files__"])
        files_ = sorted(node.get("__files__", []))
        for i, d in enumerate(dirs):
            is_last = (i == len(dirs) - 1) and not files_
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}{d}/")
            walk(node[d], prefix + ("    " if is_last else "│   "))
        for j, f in enumerate(files_):
            is_last = (j == len(files_) - 1)
            lines.append(f"{prefix}{'└── ' if is_last else '├── '}{f}")

    walk(tree)
    return "\n".join(lines)

def format_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}"
        n /= 1024.0

def write_manifest(scope: str):
    if scope == "none":
        return None
    status = git_status_map()
    scope_paths = list_tracked() if scope == "repo" else files
    stamped = time.strftime("%Y-%m-%d %H:%M:%S")
    sizes = {}
    for p in scope_paths:
        path = repo_root / p
        if path.exists() and path.is_file():
            try:
                sizes[p] = format_size(path.stat().st_size)
            except OSError:
                sizes[p] = "?"
    tree_txt = build_tree(scope_paths)
    details = []
    for p in sorted(scope_paths):
        flag = status.get(p, "  ")
        sz = sizes.get(p, "?")
        details.append(f"- `{p}`  ·  **{sz}**  ·  status: `{flag}`")
    details_block = "\n".join(details)
    md = f"""# Repository Structure

**Root:** `{repo_root.name}`  
**Generated:** {stamped}  
**Upload mode:** {"unstaged only" if args.diff else "all tracked"}  
**Structure scope:** {scope}
```
{tree_txt}
```
{details_block}
"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".md", prefix="REPO_STRUCTURE_"
    )
    tmp.write(md)
    tmp.flush()
    tmp.close()
    return tmp.name

manifest_path = write_manifest(args.structure)

vs_name = f"{repo_root.name}-review-{int(time.time())}"
vector_store = client.vector_stores.create(
    name=vs_name,
    metadata={"project": "repo-review", "repo": repo_root.name},
    expires_after={"anchor": "last_active_at", "days": 1},
)

file_paths = [str(repo_root / p) for p in files]
if manifest_path:
    file_paths.append(manifest_path)

file_streams = [open(p, "rb") for p in file_paths]
try:
    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams,
    )
finally:
    for fh in file_streams:
        try:
            fh.close()
        except Exception:
            pass
    if manifest_path:
        try:
            os.unlink(manifest_path)
        except Exception:
            pass

print(f"Vector store: {vector_store.id}  (name='{vs_name}')")
print(
    f"Batch status: {getattr(batch, 'status', 'unknown')}, counts: {getattr(batch, 'file_counts', None)}"
)

failed: List[Tuple[str, Optional[str]]] = []
files_list = client.vector_stores.files.list(vector_store_id=vector_store.id)
for f in files_list.data:
    st = getattr(f, "status", None)
    if st == "failed":
        failed.append((f.id, getattr(f, "last_error", None)))

if failed:
    print("\n[warn] Some files failed to ingest:")
    for fid, err in failed:
        print(f"  - {fid}: {err}")
    print("[warn] Continuing anyway; the model will retrieve from successfully ingested files.")

if args.prompt:
    base_prompt = args.prompt
else:
    base_prompt = (
        "Don't reference any prior conversations, files, or contexts. Only look at what is provided in this chat. Act as a world class ML researcher. This is a YOLO variant. Provide only critical needs, give me actionable items, if any, for this code base. Provide full explicit diffs. "
        "Be extremely thorough. When suggesting a change trace what relies on that change. Suggest changes to those as well. Trace dependencies to completion. Ensure all math functions are sound. Maximize thinking tokens, be as detailed as you can and return as much as you can. "
        "A manifest file describing the directory tree is included—use it to reference exact paths when helpful. "
        "**IMPORTANT** if there is no actionable item. Such as 'keep it as is' do not include it. Again, if there is no actionable item. Such as 'keep it as is' do not include it. Do not include anything that does not require a change. "
        "**IMPORTANT** Do not tell me if something looks good, or that you did a sanity check. If it looks good, do not include it. "
        "**IMPORTANT** If an actionable item is solved by another actionable item, do not include it. Again, If an actionable item is solved by another actionable item, do not include it. "
        "Each action item should be marked with 1), 2), 3) and so on. At the end of each action item, separate with the character string ©©© on a line by itself. Including the final action item. "
        "If there are no changes to be made, and the goals of my request are met, simply respond NONE"
    )
    if args.add:
        base_prompt += "\n\nAdditional instructions:\n" + args.add

content_items = [{"type": "input_text", "text": base_prompt}]

def create_response(model_id: str):
    kwargs = {
        "model": model_id,
        "input": [{
            "role": "user",
            "content": content_items,
        }],
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [vector_store.id],
        }],
    }
    try:
        return client.responses.create(**kwargs)
    except BadRequestError as e:
        raise e

try:
    resp = create_response(args.model)
except BadRequestError as e:
    msg = str(e)
    if "model_not_found" in msg or "does not exist" in msg:
        print(
            f"[warn] Model '{args.model}' not available on this account. Falling back to 'gpt-4.1'."
        )
        resp = create_response("gpt-4.1")
    else:
        raise

out = getattr(resp, "output_text", None)
if out is None:
    try:
        out = resp.output[0].content[0].text
    except Exception:
        out = str(resp)

items = _parse_action_items(out or "")
count = _write_action_plan(items, repo_root)

print(out)
print(f"[info] Wrote {count} action item file(s) to '{ACTION_PLAN_DIRNAME}/' as 1.txt..{count}.txt")
