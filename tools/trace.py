#!/usr/bin/env python3

import argparse
import ast
import sys
from pathlib import Path
import importlib.util
from typing import Dict, Set, List, Optional, Tuple

def find_package_root(entry_file: Path) -> Path:
    """
    Return the package/project root: the highest directory *below* the first
    parent that doesn't contain an __init__.py. If none contain it, use the
    entry file's parent.
    """
    entry_file = entry_file.resolve()
    p = entry_file.parent
    last_pkg = p
    while (p / "__init__.py").exists():
        last_pkg = p
        p = p.parent
    return last_pkg

def module_name_from_path(path: Path, roots: List[Path]) -> Optional[str]:
    """
    Best-effort module name from a filesystem path, relative to any of `roots`.
    """
    path = path.resolve()
    for root in roots:
        root = root.resolve()
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = list(rel.parts)
        if not parts:
            return None
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        else:
            return None
        parts = [p for p in parts if p]
        return ".".join(parts)
    return None

def is_python_source_spec(spec) -> bool:
    """
    True if the module spec points to a .py file (module or package __init__).
    """
    if spec is None:
        return False
    origin = getattr(spec, "origin", None)
    if not origin or origin == "built-in":
        return False
    return origin.endswith(".py")

def in_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

def _is_main_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    left = test.left
    rights = test.comparators
    if len(rights) != 1:
        return False

    def _is_name(node, name):
        return isinstance(node, ast.Name) and node.id == name

    def _is_str(node, s):
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == s

    return (_is_name(left, "__name__") and _is_str(rights[0], "__main__")) or \
           (_is_str(left, "__main__") and _is_name(rights[0], "__name__"))

def collect_imports(mod: ast.Module, current_modname: str) -> Set[Tuple[str, Optional[str], int]]:
    """
    Return a set of tuples describing imports:
        (base_module, imported_name_or_None, level)
    For `import x.y`, we record ('x.y', None, 0).
    For `from x import y`, we record ('x', 'y', 0).
    For relative imports, level > 0 and base_module may be '' for purely relative.
    """
    results = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.Import):
            for a in node.names:
                results.add((a.name, None, 0))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            results.add((
                base, None if any(n.name == "*" for n in node.names) else "_SENTINEL_", node.level
            ))
            if not any(n.name == "*" for n in node.names):
                for a in node.names:
                    results.add((base, a.name, node.level))
    return results

def resolve_relative_name(current_modname: str, module: str, level: int) -> Optional[str]:
    """
    Resolve a relative import base to an absolute dotted module name.
    """
    if level == 0:
        return module or ""
    parts = current_modname.split(".") if current_modname else []
    if level > len(parts):
        return None
    base = ".".join(parts[:-level])
    if module:
        return f"{base}.{module}" if base else module
    return base

class CleanAndKeepDefs(ast.NodeTransformer):
    """
    Keep only top-level ClassDef, FunctionDef, AsyncFunctionDef.
    Strip module/class/function docstrings.
    Optionally remove decorators (which otherwise may require missing imports).
    """
    def __init__(self, drop_decorators: bool = False):
        self.drop_decorators = drop_decorators
        super().__init__()

    @staticmethod
    def _strip_leading_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
        if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant
        ) and isinstance(body[0].value.value, str):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._strip_leading_docstring(node.body)
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.If) and _is_main_guard(stmt.test):
                continue  # drop if __name__ == "__main__"
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                new_body.append(stmt)
        node.body = [self.visit(s) for s in new_body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._strip_leading_docstring(node.body)
        node.type_comment = None
        if self.drop_decorators:
            node.decorator_list = []
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._strip_leading_docstring(node.body)
        node.type_comment = None
        if self.drop_decorators:
            node.decorator_list = []
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = self._strip_leading_docstring(node.body)
        if self.drop_decorators:
            node.decorator_list = []
        return self.generic_visit(node)

class ImportTracer:
    def __init__(self, root: Path, follow_external: bool = False):
        self.root = root.resolve()
        self.follow_external = follow_external
        self.visited: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}
        self.mod_to_path: Dict[str, Path] = {}

    def _spec_ok(self, spec) -> bool:
        if not is_python_source_spec(spec):
            return False
        if self.follow_external:
            return True
        origin_path = Path(spec.origin)
        return in_root(origin_path, self.root)

    def _find_spec_path(self, modname: str) -> Optional[Path]:
        try:
            spec = importlib.util.find_spec(modname)
        except (ModuleNotFoundError, ValueError):
            return None
        if spec and self._spec_ok(spec):
            return Path(spec.origin).resolve()
        return None

    def _parse(self, path: Path) -> Optional[ast.Module]:
        try:
            src = path.read_text(encoding="utf-8")
        except Exception:
            return None
        try:
            return ast.parse(src, filename=str(path))
        except SyntaxError:
            return None

    def trace_from(self, entry_path: Path, entry_modname: str) -> List[str]:
        """
        DFS trace; returns a topological-ish order list of module names.
        """
        sys_path_inserted = False
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
            sys_path_inserted = True

        order: List[str] = []

        def dfs(modname: str):
            if modname in self.visited:
                return
            self.visited.add(modname)

            path = self._find_spec_path(modname)
            if not path:
                return
            self.mod_to_path[modname] = path

            mod_ast = self._parse(path)
            if not mod_ast:
                return

            imps = collect_imports(mod_ast, modname)
            children: Set[str] = set()
            for base, name, level in imps:
                abs_base = resolve_relative_name(modname, base, level)
                if abs_base is None:
                    continue
                candidates = []

                if name is None:
                    candidates.append(abs_base)
                elif name == "_SENTINEL_":
                    pass
                else:
                    candidates.append(f"{abs_base}.{name}")
                    candidates.append(abs_base)

                for cand in candidates:
                    p = self._find_spec_path(cand)
                    if p:
                        children.add(cand)

            self.edges[modname] = children
            for c in sorted(children):
                dfs(c)

            order.append(modname)

        try:
            dfs(entry_modname)
        finally:
            if sys_path_inserted:
                try:
                    sys.path.remove(str(self.root))
                except ValueError:
                    pass

        return order[::-1]

def extract_clean_code(path: Path, drop_decorators: bool) -> str:
    src = path.read_text(encoding="utf-8")
    mod = ast.parse(src, filename=str(path))
    mod = CleanAndKeepDefs(drop_decorators=drop_decorators).visit(mod)
    ast.fix_missing_locations(mod)
    return ast.unparse(mod).strip()

def main():
    parser = argparse.ArgumentParser(
        description=
        "Trace an entry module's imports and bundle all classes/functions into one file with no docstrings."
    )
    parser.add_argument("entry", help="Path to entry .py file")
    parser.add_argument("output", help="Path to output .py file")
    parser.add_argument("--project-root", help="Project root (defaults to package root of entry)")
    parser.add_argument(
        "--follow-external",
        action="store_true",
        help="Also include modules outside project root (site-packages etc.) if they are .py"
    )
    parser.add_argument(
        "--drop-decorators",
        action="store_true",
        help="Remove all decorators from functions/methods/classes"
    )
    args = parser.parse_args()

    entry_path = Path(args.entry).resolve()
    if not entry_path.exists():
        print(f"Entry file not found: {entry_path}", file=sys.stderr)
        sys.exit(1)

    root = Path(args.project_root).resolve() if args.project_root else find_package_root(entry_path)
    entry_modname = module_name_from_path(entry_path, [root])
    if not entry_modname:
        entry_modname = entry_path.stem

    tracer = ImportTracer(root=root, follow_external=args.follow_external)
    order = tracer.trace_from(entry_path, entry_modname)

    if entry_modname not in tracer.mod_to_path:
        tracer.mod_to_path[entry_modname] = entry_path
        if entry_modname not in order:
            order.insert(0, entry_modname)

    blocks: List[str] = []
    seen_files: Set[Path] = set()

    for modname in order:
        path = tracer.mod_to_path.get(modname)
        if not path or path in seen_files:
            continue
        seen_files.add(path)
        try:
            code = extract_clean_code(path, drop_decorators=args.drop_decorators)
        except Exception:
            continue
        if code:
            blocks.append(code)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    final = ("\n\n".join(b for b in blocks if b.strip()) + "\n").lstrip()
    output.write_text(final, encoding="utf-8")

if __name__ == "__main__":
    main()
