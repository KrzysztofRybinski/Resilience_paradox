"""Fixed-effects regressions for main specifications."""
from __future__ import annotations

import pandas as pd
from linearmodels.panel import PanelOLS

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet
from resilience_paradox.models.tables import write_regression_table


def _prepare_panel(paths: Paths) -> pd.DataFrame:
    df = read_parquet(paths.data_final / "panel_annual.parquet")
    df = df.dropna(subset=["dln_gross_output"])
    df = df.set_index(["country_iso3", "icio50", "year"]).sort_index()
    return df


def estimate_main(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_csv = paths.output / "tables" / "Table2_baseline_effects.csv"
    output_tex = paths.output / "tables" / "Table2_baseline_effects.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Baseline estimates already exist; skipping.")
        return

    df = _prepare_panel(paths)
    y = df["dln_gross_output"]

    clusters = df.reset_index()[["country_iso3", "year"]]

    model1 = PanelOLS(y, df[["exposure_total"]], entity_effects=True, time_effects=True)
    res1 = model1.fit(cov_type="clustered", clusters=clusters)

    model2 = PanelOLS(
        y,
        df[["exposure_total", "exposure_conc"]],
        entity_effects=True,
        time_effects=True,
    )
    res2 = model2.fit(cov_type="clustered", clusters=clusters)

    write_regression_table(
        {"Baseline": res1, "Concentration": res2},
        output_csv,
        output_tex,
    )
    logger.info("Wrote baseline regression table")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_main",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )


def estimate_shock(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_csv = paths.output / "tables" / "Table3_shock_interactions.csv"
    output_tex = paths.output / "tables" / "Table3_shock_interactions.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Shock interaction estimates already exist; skipping.")
        return

    df = _prepare_panel(paths)
    df = df.reset_index()
    df["exposure_covid"] = df["exposure_total"] * df["post_covid"]
    df["exposure_energy"] = df["exposure_total"] * df["shock_energy"]
    df = df.set_index(["country_iso3", "icio50", "year"]).sort_index()

    y = df["dln_gross_output"]
    clusters = df.reset_index()[["country_iso3", "year"]]

    model = PanelOLS(
        y,
        df[["exposure_covid", "exposure_energy"]],
        entity_effects=True,
        time_effects=True,
    )
    res = model.fit(cov_type="clustered", clusters=clusters)

    write_regression_table({"Shock interactions": res}, output_csv, output_tex)
    logger.info("Wrote shock interaction table")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_shock",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )
