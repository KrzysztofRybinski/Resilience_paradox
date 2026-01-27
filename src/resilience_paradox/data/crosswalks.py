"""Crosswalks for NACE to ICIO50 mapping."""
from __future__ import annotations

import pandas as pd

from resilience_paradox.paths import Paths
from resilience_paradox.utils.io import read_parquet


SPLIT_MAP = {
    "24": {"C24A": 0.5, "C24B": 0.5},
    "30": {"C301": 0.5, "C302T309": 0.5},
}


def load_split_weights(paths: Paths) -> dict[str, dict[str, float]]:
    path = paths.data_int / "icio_split_weights.parquet"
    if path.exists():
        df = read_parquet(path)
        weights = {}
        for _, row in df.iterrows():
            key = row["nace_code"]
            weights.setdefault(key, {})[row["icio50"]] = row["weight"]
        return weights
    return SPLIT_MAP


def _load_mapping(paths: Paths) -> dict[str, str]:
    mapping_path = paths.resolve_project_path("config/nace_to_icio50.csv")
    if not mapping_path.exists():
        return {}
    mapping_df = pd.read_csv(mapping_path, dtype=str)
    return dict(zip(mapping_df["nace_code"], mapping_df["icio50"]))


def map_nace_to_icio50(
    df: pd.DataFrame, weights: dict[str, dict[str, float]], paths: Paths
) -> pd.DataFrame:
    mapping = _load_mapping(paths)
    records = []
    for _, row in df.iterrows():
        nace = str(row.get("nace_code", "")).strip()
        key = nace.split(".")[0] if nace else ""
        if key in weights:
            for icio, share in weights[key].items():
                new_row = row.copy()
                new_row["icio50"] = icio
                new_row["aid_amount_eur"] = row["aid_amount_eur"] * share
                records.append(new_row)
        elif nace in mapping:
            new_row = row.copy()
            new_row["icio50"] = mapping[nace]
            records.append(new_row)
        else:
            new_row = row.copy()
            new_row["icio50"] = f"C{key}" if key else "UNKNOWN"
            records.append(new_row)
    return pd.DataFrame(records)
