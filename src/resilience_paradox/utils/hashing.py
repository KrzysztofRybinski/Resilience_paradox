"""Hashing utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dict_hash(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()
