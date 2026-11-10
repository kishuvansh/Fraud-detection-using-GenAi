"""Small I/O helpers: config loading, file hashing and JSON that understands NumPy types."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(root: Path, name: str = "data") -> dict:
    with open(root / "configs" / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def sha256_array(values: np.ndarray) -> str:
    """Stable fingerprint of an integer index array (used to prove splits are reproducible)."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def save_json(obj: Any, path: Path, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_to_builtin)
        fh.write("\n")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
