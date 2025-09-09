#!/usr/bin/env python3
"""
tools/dump_config.py

Dump the active training config (including fully merged hyp) from a checkpoint
file or run directory. If a config.yaml exists alongside checkpoints, it is
used directly. Otherwise, the script reconstructs the active hyp by deep-merging
defaults with the checkpoint's embedded cfg via core.config.get_config.

Usage:
  python tools/dump_config.py <path> [-o out.yaml] [--hyp-only]

Where <path> is one of:
  - A checkpoint file (e.g., runs/exp/checkpoints/last.pt)
  - A run directory (e.g., runs/exp). The script will prefer
    runs/exp/checkpoints/config.yaml, else inspect last.pt/best.pt.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

def _load_yaml_file(p: Path) -> dict:
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def _dump_yaml(data: dict, out: Path | None) -> None:
    txt = yaml.safe_dump(data, sort_keys=False)
    if out is None:
        sys.stdout.write(txt)
        sys.stdout.flush()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(txt)

def _find_ckpt_files(run_dir: Path) -> list[Path]:
    cdir = run_dir / 'checkpoints'
    files: list[Path] = []
    for name in ('last.pt', 'best.pt'):
        p = cdir / name
        if p.is_file():
            files.append(p)
    return files

def dump_from_checkpoint_path(path: Path, out: Path | None, hyp_only: bool) -> int:
    if path.is_dir():
        cfg_yml = path / 'checkpoints' / 'config.yaml'
        if cfg_yml.is_file():
            data = _load_yaml_file(cfg_yml)
            data = data.get('hyp', {}) if hyp_only else data
            _dump_yaml(data, out)
            return 0
        cfg_yml2 = path / 'config.yaml'
        if cfg_yml2.is_file():
            data = _load_yaml_file(cfg_yml2)
            data = data.get('hyp', {}) if hyp_only else data
            _dump_yaml(data, out)
            return 0
        cand = _find_ckpt_files(path)
        if not cand:
            sys.stderr.write(f"No checkpoints or config.yaml found under {path}\n")
            return 2
        path = cand[0]

    try:
        import torch  # local import to avoid hard dep if not needed
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        sys.stderr.write(f"Failed to load checkpoint: {e}\n")
        return 3

    cfg = ckpt.get('cfg', {}) if isinstance(ckpt, dict) else {}
    if not isinstance(cfg, dict) or not cfg:
        sys.stderr.write("Checkpoint missing cfg block; cannot reconstruct config.\n")
        return 4

    try:
        from core.config import get_config
        active = get_config(cfg=cfg)
        cfg_out = dict(cfg)
        cfg_out['hyp'] = dict(active.hyp) if hasattr(active, 'hyp') else cfg.get('hyp', {})
    except Exception:
        cfg_out = dict(cfg)

    data = cfg_out.get('hyp', {}) if hyp_only else cfg_out
    _dump_yaml(data, out)
    return 0

def main():
    ap = argparse.ArgumentParser(description='Dump active config from checkpoint or run dir')
    ap.add_argument('path', type=str, help='Path to checkpoint file (.pt) or run directory')
    ap.add_argument(
        '-o', '--out', type=str, default=None, help='Output YAML file (default: stdout)'
    )
    ap.add_argument('--hyp-only', action='store_true', help='Dump only the hyp section')
    args = ap.parse_args()

    p = Path(args.path)
    out = Path(args.out) if args.out else None
    code = dump_from_checkpoint_path(p, out, args.hyp_only)
    sys.exit(code)

if __name__ == '__main__':
    main()
