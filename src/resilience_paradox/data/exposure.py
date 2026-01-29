"""Exposure and final panel construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from resilience_paradox.config import AppConfig, load_shocks
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_parquet


def build_exposure(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_path = paths.data_final / "exposure_panel.parquet"
    if output_path.exists() and not force:
        logger.info("Exposure panel exists; skipping.")
        return

    aid_panel = read_parquet(paths.data_final / "upstream_aid_panel.parquet")
    output_va = read_parquet(paths.data_int / "icio_output_va.parquet")
    shares = read_parquet(paths.data_int / "icio_input_shares_base.parquet")

    if sample:
        output_va["usd_per_eur"] = 1.0
        output_va["deflator_to_base"] = 1.0
    else:
        from resilience_paradox.data.prices import load_fx_annual, load_hicp_deflator

        fx = load_fx_annual(paths, config)
        usd = fx[fx["currency"] == "USD"][["year", "rate_per_eur"]].rename(
            columns={"rate_per_eur": "usd_per_eur"}
        )
        hicp = load_hicp_deflator(paths, config)[["year", "deflator_to_base"]]

        output_va = output_va.merge(usd, on="year", how="left").merge(hicp, on="year", how="left")
        if output_va[["usd_per_eur", "deflator_to_base"]].isna().any().any():
            raise RuntimeError(
                "Missing USD/EUR FX rates or HICP deflators for some ICIO years. "
                "Run `rp prices download` (or rerun with --force) to refresh cached price tables."
            )
    output_va["gross_output_eur_million"] = output_va["gross_output"] / output_va["usd_per_eur"]
    output_va["gross_output_real_eur_million"] = (
        output_va["gross_output_eur_million"] * output_va["deflator_to_base"]
    )

    merged = aid_panel.merge(
        output_va,
        left_on=["country_iso3", "icio50", "year"],
        right_on=["country_iso3", "icio50", "year"],
        how="left",
    )
    merged["subsidy_intensity"] = merged["aid_total_real_eur_million"] / merged[
        "gross_output_real_eur_million"
    ]
    merged = merged.rename(columns={"icio50": "upstream_icio50"})

    exposure = shares.merge(
        merged,
        on=["country_iso3", "upstream_icio50"],
        how="left",
    )
    exposure["exposure_total"] = exposure["ioshare_base"] * exposure["subsidy_intensity"]
    exposure["exposure_conc"] = exposure["ioshare_base"] * exposure["hhi_beneficiary"]
    exposure["exposure_aidlevel"] = exposure["ioshare_base"] * exposure["aid_total_real_eur_million"]

    grouped = (
        exposure.groupby(["country_iso3", "downstream_icio50", "year"], as_index=False)
        .agg(
            exposure_total=("exposure_total", "sum"),
            exposure_conc=("exposure_conc", "sum"),
            exposure_aidlevel=("exposure_aidlevel", "sum"),
        )
        .rename(columns={"downstream_icio50": "icio50"})
    )

    shocks = load_shocks(paths.resolve_project_path("config/shocks.toml"))
    grouped["post_covid"] = grouped["year"] >= shocks["covid"]["start"]
    grouped["shock_energy"] = grouped["year"].between(
        shocks["energy"]["start"], shocks["energy"]["end"]
    )

    write_parquet(grouped, output_path)
    logger.info("Wrote exposure panel to %s", output_path)
    record_manifest(
        paths,
        config.model_dump(),
        "build_exposure",
        [paths.data_final / "upstream_aid_panel.parquet"],
        [output_path],
    )


def build_panel(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_path = paths.data_final / "panel_annual.parquet"
    if output_path.exists() and not force:
        logger.info("Final panel exists; skipping.")
        return

    output_va = read_parquet(paths.data_int / "icio_output_va.parquet").rename(
        columns={"gross_output": "gross_output_usd_million", "value_added": "value_added_usd_million"}
    )
    exposure = read_parquet(paths.data_final / "exposure_panel.parquet")
    hhi = read_parquet(paths.data_int / "icio_import_hhi.parquet")

    if sample:
        df = output_va.copy()
        df["usd_per_eur"] = 1.0
        df["deflator_to_base"] = 1.0
    else:
        from resilience_paradox.data.prices import load_fx_annual, load_hicp_deflator

        fx = load_fx_annual(paths, config)
        usd = fx[fx["currency"] == "USD"][["year", "rate_per_eur"]].rename(
            columns={"rate_per_eur": "usd_per_eur"}
        )
        hicp = load_hicp_deflator(paths, config)[["year", "deflator_to_base"]]

        df = output_va.merge(usd, on="year", how="left").merge(hicp, on="year", how="left")
        if df[["usd_per_eur", "deflator_to_base"]].isna().any().any():
            raise RuntimeError(
                "Missing USD/EUR FX rates or HICP deflators for some years. "
                "Run `rp prices download` (or rerun with --force) to refresh cached price tables."
            )
    df["gross_output"] = (df["gross_output_usd_million"] / df["usd_per_eur"]) * df["deflator_to_base"]
    df["value_added"] = (df["value_added_usd_million"] / df["usd_per_eur"]) * df["deflator_to_base"]

    df = df.merge(exposure, on=["country_iso3", "icio50", "year"], how="left")
    df = df.merge(hhi, on=["country_iso3", "icio50", "year"], how="left")

    df["ln_gross_output"] = df["gross_output"].apply(
        lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
    )
    df = df.sort_values(["country_iso3", "icio50", "year"])
    df["dln_gross_output"] = df.groupby(["country_iso3", "icio50"])["ln_gross_output"].diff()
    df["ln_value_added"] = df["value_added"].apply(
        lambda x: pd.NA if pd.isna(x) or x <= 0 else np.log(x)
    )
    df["dln_value_added"] = df.groupby(["country_iso3", "icio50"])["ln_value_added"].diff()

    shocks = load_shocks(paths.resolve_project_path("config/shocks.toml"))
    df["post_covid"] = df["year"] >= shocks["covid"]["start"]
    df["shock_energy"] = df["year"].between(shocks["energy"]["start"], shocks["energy"]["end"])

    write_parquet(df, output_path)
    logger.info("Wrote final panel to %s", output_path)
    record_manifest(
        paths,
        config.model_dump(),
        "build_panel",
        [paths.data_final / "exposure_panel.parquet"],
        [output_path],
    )
