"""IO helpers for parquet/CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def list_files(path: Path, suffix: str) -> Iterable[Path]:
    return sorted(path.rglob(f"*{suffix}"))
