"""Clean State Aid awards into standardized parquet."""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import write_parquet

COLUMN_MAP = {
    "Member state": "country_iso3",
    "Granting date": "granting_date",
    "Beneficiary": "beneficiary_name",
    "Aid element (EUR)": "aid_amount_eur",
    "Aid instrument": "aid_instrument",
    "Aid objective": "aid_objective",
    "NACE code": "nace_code",
    "NUTS2": "nuts2",
    "Measure ID": "measure_id",
    "Case ID": "case_id",
}


def _hash_row(row: pd.Series) -> str:
    payload = "|".join(str(value) for value in row.values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_amount(value: str | float | int) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(" ", "").replace(",", "")
    if "-" in cleaned or "–" in cleaned:
        parts = re.split(r"[-–]", cleaned)
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_stateaid(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_path = paths.data_int / "stateaid_awards.parquet"
    if output_path.exists() and not force:
        logger.info("State aid awards already cleaned; skipping.")
        return

    raw_dir = paths.data_raw / "stateaid"
    frames = []
    for csv_path in raw_dir.rglob("*.csv"):
        df = pd.read_csv(csv_path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No state aid CSV files found. Run rp stateaid download first.")
    raw = pd.concat(frames, ignore_index=True)

    df = raw.rename(columns=COLUMN_MAP)
    for col in COLUMN_MAP.values():
        if col not in df.columns:
            df[col] = np.nan
    df["granting_date"] = pd.to_datetime(df["granting_date"], errors="coerce")
    df["year"] = df["granting_date"].dt.year
    df["aid_amount_eur"] = df["aid_amount_eur"].apply(_parse_amount)
    df = df.dropna(subset=["country_iso3", "aid_amount_eur"])
    df["award_id"] = df.apply(_hash_row, axis=1)
    keep_cols = [
        "award_id",
        "country_iso3",
        "granting_date",
        "year",
        "beneficiary_name",
        "aid_amount_eur",
        "aid_instrument",
        "aid_objective",
        "nace_code",
        "nuts2",
        "measure_id",
        "case_id",
    ]
    df = df[keep_cols]
    write_parquet(df, output_path)
    logger.info("Wrote cleaned state aid awards to %s", output_path)
    record_manifest(
        paths,
        config.model_dump(),
        "stateaid_clean",
        [paths.data_raw / "stateaid"],
        [output_path],
    )
