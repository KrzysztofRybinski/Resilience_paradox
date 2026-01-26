"""Efficient ICIO reader utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl


def read_icio_long(path: Path, years: Iterable[int]) -> pl.LazyFrame:
    """Read ICIO long-format CSV with minimal columns."""
    scan = pl.scan_csv(path)
    return scan.filter(pl.col("year").is_in(list(years)))


def compute_output_va(df: pl.DataFrame) -> pd.DataFrame:
    output = (
        df.filter(pl.col("flow_type") == "go")
        .group_by(["origin_country", "origin_sector", "year"])
        .agg(pl.col("value").sum().alias("gross_output"))
    )
    va = (
        df.filter(pl.col("flow_type") == "va")
        .group_by(["origin_country", "origin_sector", "year"])
        .agg(pl.col("value").sum().alias("value_added"))
    )
    merged = output.join(va, on=["origin_country", "origin_sector", "year"], how="left")
    return merged.rename(
        {
            "origin_country": "country_iso3",
            "origin_sector": "icio50",
        }
    ).to_pandas()


def compute_input_shares_base(df: pl.DataFrame, base_years: Iterable[int]) -> pd.DataFrame:
    base_df = df.filter(
        (pl.col("flow_type") == "intermediate") & pl.col("year").is_in(list(base_years))
    )
    totals = (
        base_df.group_by(["dest_country", "dest_sector", "year"])
        .agg(pl.col("value").sum().alias("total_inputs"))
    )
    by_sector = (
        base_df.group_by(["dest_country", "dest_sector", "origin_sector", "year"])
        .agg(pl.col("value").sum().alias("sector_inputs"))
    )
    shares = by_sector.join(totals, on=["dest_country", "dest_sector", "year"], how="left")
    shares = shares.with_columns(
        (pl.col("sector_inputs") / pl.col("total_inputs")).alias("ioshare")
    )
    avg = (
        shares.group_by(["dest_country", "dest_sector", "origin_sector"])
        .agg(pl.col("ioshare").mean().alias("ioshare_base"))
    )
    return avg.rename(
        {
            "dest_country": "country_iso3",
            "dest_sector": "downstream_icio50",
            "origin_sector": "upstream_icio50",
        }
    ).to_pandas()


def compute_import_hhi(df: pl.DataFrame) -> pd.DataFrame:
    interm = df.filter(pl.col("flow_type") == "intermediate")
    totals = (
        interm.group_by(["dest_country", "dest_sector", "year"])
        .agg(pl.col("value").sum().alias("total_inputs"))
    )
    origin = (
        interm.group_by(["dest_country", "dest_sector", "origin_country", "year"])
        .agg(pl.col("value").sum().alias("origin_inputs"))
    )
    merged = origin.join(totals, on=["dest_country", "dest_sector", "year"], how="left")
    merged = merged.with_columns((pl.col("origin_inputs") / pl.col("total_inputs")).alias("share"))
    merged = merged.with_columns(
        (pl.when(pl.col("origin_country") != pl.col("dest_country"))
        .then(pl.col("origin_inputs"))
        .otherwise(0.0))
        .alias("import_inputs")
    )
    hhi = (
        merged.group_by(["dest_country", "dest_sector", "year"])
        .agg(
            (pl.col("share") ** 2).sum().alias("import_hhi_intermediate"),
            (pl.col("import_inputs").sum() / pl.col("total_inputs").first()).alias(
                "import_share_intermediate"
            ),
        )
    )
    return hhi.rename(
        {
            "dest_country": "country_iso3",
            "dest_sector": "icio50",
        }
    ).to_pandas()
