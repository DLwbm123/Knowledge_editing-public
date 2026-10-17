"""Scope-limited byte accounting with headroom for atomic saves and snapshots."""
import os
from pathlib import Path

LIMIT = 9_000_000_000


def owned_bytes(root):
    total = 0
    for folder,dirs,files in os.walk(root,followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in files:
            p = Path(folder)/name
            if not p.is_symlink():
                total += p.stat().st_size
    return total


def ensure(root,additional=0):
    size = owned_bytes(root)
    if size+additional >= LIMIT:
        raise OSError('New-run storage reservation would reach 9 GB; preserve active state')
    return size
