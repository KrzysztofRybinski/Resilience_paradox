"""Build ICIO panels for output/VA, base IO shares, and import HHI."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import polars as pl

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import write_parquet
from resilience_paradox.data.oecd_icio_reader import (
    read_icio_long,
    compute_output_va,
    compute_input_shares_base,
    compute_import_hhi,
)


def _extract_first_csv(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV files found inside {zip_path}")
        target = extract_dir / Path(members[0]).name
        if not target.exists():
            archive.extract(members[0], extract_dir)
            extracted = extract_dir / members[0]
            extracted.rename(target)
    return target


def build_icio_panels(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    output_path = paths.data_int / "icio_output_va.parquet"
    shares_path = paths.data_int / "icio_input_shares_base.parquet"
    hhi_path = paths.data_int / "icio_import_hhi.parquet"

    split_path = paths.data_int / "icio_split_weights.parquet"
    if (
        not force
        and output_path.exists()
        and shares_path.exists()
        and hhi_path.exists()
        and split_path.exists()
    ):
        logger.info("ICIO panels already exist; skipping.")
        return

    if sample:
        logger.info("Building sample ICIO panels")
        df = pl.DataFrame(
            {
                "year": [2013, 2013, 2016, 2016],
                "origin_country": ["FRA", "DEU", "FRA", "DEU"],
                "origin_sector": ["C24A", "C24A", "C24A", "C24A"],
                "dest_country": ["FRA", "FRA", "FRA", "FRA"],
                "dest_sector": ["C24A", "C24A", "C24A", "C24A"],
                "flow_type": ["intermediate", "intermediate", "go", "va"],
                "value": [100.0, 50.0, 500.0, 200.0],
            }
        )
        output_va = compute_output_va(df)
        shares = compute_input_shares_base(df, range(config.years.base_start, config.years.base_end + 1))
        hhi = compute_import_hhi(df)
    else:
        raw_dir = paths.oecd_raw_dir()
        bundle = config.oecd.icio.bundles[0]
        zip_path = raw_dir / f"ICIO{config.oecd.icio.release}_{bundle}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Missing {zip_path}. Download with `rp oecd download` or place manually."
            )
        csv_path = _extract_first_csv(zip_path, raw_dir / "extracted")
        years = range(config.years.base_start, config.years.end + 1)
        lazy = read_icio_long(csv_path, years)
        df = lazy.collect()
        output_va = compute_output_va(df)
        shares = compute_input_shares_base(df, range(config.years.base_start, config.years.base_end + 1))
        hhi = compute_import_hhi(df)

    output_va_df = output_va.copy()
    base_years = range(config.years.base_start, config.years.base_end + 1)
    base_va = output_va_df[output_va_df["year"].isin(base_years)]
    split_rows = []
    for nace_code, icio_list in {"24": ["C24A", "C24B"], "30": ["C301", "C302T309"]}.items():
        subset = base_va[base_va["icio50"].isin(icio_list)]
        total = subset["gross_output"].sum()
        for icio in icio_list:
            weight = 0.5 if total == 0 else subset.loc[subset["icio50"] == icio, "gross_output"].sum() / total
            split_rows.append({"nace_code": nace_code, "icio50": icio, "weight": weight})
    split_weights = pd.DataFrame(split_rows)

    write_parquet(output_va, output_path)
    write_parquet(shares, shares_path)
    write_parquet(hhi, hhi_path)
    write_parquet(split_weights, split_path)
    logger.info("Wrote ICIO panels to %s", paths.data_int)
    record_manifest(
        paths,
        config.model_dump(),
        "oecd_build_icio_panels",
        [paths.data_raw / "oecd_icio"],
        [output_path, shares_path, hhi_path, split_path],
    )
