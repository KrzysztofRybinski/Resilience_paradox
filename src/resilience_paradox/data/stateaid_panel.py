"""Build upstream aid panel by country-sector-year."""
from __future__ import annotations

import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.data.crosswalks import map_nace_to_icio50, load_split_weights
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_parquet
from resilience_paradox.utils.pandas_helpers import gini, hhi


def build_upstream_aid_panel(
    config: AppConfig, force: bool = False, sample: bool = False
) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_path = paths.data_final / "upstream_aid_panel.parquet"
    if output_path.exists() and not force:
        logger.info("Upstream aid panel exists; skipping.")
        return

    awards_path = paths.data_int / "stateaid_awards.parquet"
    df = read_parquet(awards_path)

    split_weights = load_split_weights(paths)

    mapped = map_nace_to_icio50(df, split_weights, paths)

    if "aid_amount_eur_million" in mapped.columns:
        mapped["aid_amount_eur_million"] = mapped["aid_amount_eur_million"]
    else:
        mapped["aid_amount_eur_million"] = mapped["aid_amount_eur"] / 1e6

    if "aid_amount_real_eur_million" not in mapped.columns:
        mapped["aid_amount_real_eur_million"] = mapped["aid_amount_eur_million"]

    grouped = (
        mapped.groupby(["country_iso3", "icio50", "year"], as_index=False)
        .agg(
            aid_total_eur_million=("aid_amount_eur_million", "sum"),
            aid_total_real_eur_million=("aid_amount_real_eur_million", "sum"),
            n_awards=("award_id", "nunique"),
            n_beneficiaries=("beneficiary_name", "nunique"),
        )
        .copy()
    )

    beneficiary_stats = (
        mapped.groupby(["country_iso3", "icio50", "year", "beneficiary_name"], as_index=False)
        .agg(amount=("aid_amount_real_eur_million", "sum"))
    )
    concentration = (
        beneficiary_stats.groupby(["country_iso3", "icio50", "year"])  # type: ignore
        .apply(
            lambda g: pd.Series(
                {
                    "share_top20_beneficiaries": g.sort_values("amount", ascending=False)
                    .head(max(1, int(len(g) * 0.2)))["amount"].sum()
                    / g["amount"].sum(),
                    "hhi_beneficiary": hhi(g["amount"]),
                    "gini_beneficiary": gini(g["amount"]),
                }
            )
        )
        .reset_index()
    )

    merged = grouped.merge(concentration, on=["country_iso3", "icio50", "year"], how="left")
    write_parquet(merged, output_path)
    logger.info("Wrote upstream aid panel to %s", output_path)
    record_manifest(
        paths,
        config.model_dump(),
        "stateaid_build_panel",
        [paths.data_int / "stateaid_awards.parquet"],
        [output_path],
    )
