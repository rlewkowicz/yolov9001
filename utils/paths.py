from pathlib import Path

def increment_path(path: str | Path, exist_ok: bool = False) -> Path:
    path = Path(path)
    if exist_ok or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    base, name = path.parent, path.name
    i = 1
    while (base / f"{name}{i}").exists():
        i += 1
    run_dir = base / f"{name}{i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
