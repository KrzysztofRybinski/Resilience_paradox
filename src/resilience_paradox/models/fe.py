"""Fixed-effects regressions for main specifications."""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet
from resilience_paradox.models.tables import write_regression_table


def _prepare_panel(paths: Paths, config: AppConfig) -> pd.DataFrame:
    df = read_parquet(paths.data_final / "panel_annual.parquet")
    df = df[df["year"].between(config.years.start, config.years.end)]
    df = df.dropna(subset=["dln_gross_output"])
    df["entity_id"] = df["country_iso3"].astype(str) + "_" + df["icio50"].astype(str)
    df = df.set_index(["entity_id", "year"]).sort_index()
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

    df = _prepare_panel(paths, config)
    df = df.dropna(subset=["dln_gross_output", "exposure_total", "exposure_conc"])
    y = df["dln_gross_output"]

    clusters = pd.DataFrame(
        {
            "country_iso3": df["country_iso3"].astype(str),
            "year": df.index.get_level_values("year"),
        },
        index=df.index,
    )

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

    df = _prepare_panel(paths, config)
    df = df.dropna(subset=["dln_gross_output", "exposure_total", "post_covid", "shock_energy"])
    df = df.reset_index()
    df["exposure_covid"] = df["exposure_total"] * df["post_covid"]
    df["exposure_energy"] = df["exposure_total"] * df["shock_energy"]
    df = df.set_index(["entity_id", "year"]).sort_index()

    y = df["dln_gross_output"]
    clusters = pd.DataFrame(
        {
            "country_iso3": df["country_iso3"].astype(str),
            "year": df.index.get_level_values("year"),
        },
        index=df.index,
    )

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


def estimate_robustness(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    """Robustness checks for the baseline specification.

    Writes:
      - output/tables/Table4_robustness.csv
      - output/tables/Table4_robustness.tex
    """

    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_csv = paths.output / "tables" / "Table4_robustness.csv"
    output_tex = paths.output / "tables" / "Table4_robustness.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Robustness estimates already exist; skipping.")
        return

    df = _prepare_panel(paths, config)

    def make_clusters(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country_iso3": frame["country_iso3"].astype(str),
                "year": frame.index.get_level_values("year"),
            },
            index=frame.index,
        )

    results: dict[str, object] = {}

    base = df.dropna(subset=["dln_gross_output", "exposure_total"])
    y = base["dln_gross_output"]
    clusters = make_clusters(base)
    res = PanelOLS(y, base[["exposure_total"]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", clusters=clusters
    )
    results["Baseline"] = res

    no_2016 = base[base.index.get_level_values("year") != 2016]
    if not no_2016.empty:
        y = no_2016["dln_gross_output"]
        clusters = make_clusters(no_2016)
        res = PanelOLS(
            y, no_2016[["exposure_total"]], entity_effects=True, time_effects=True
        ).fit(cov_type="clustered", clusters=clusters)
        results["Drop 2016"] = res

    conc = df.dropna(subset=["dln_gross_output", "exposure_total", "exposure_conc"])
    y = conc["dln_gross_output"]
    clusters = make_clusters(conc)
    res = PanelOLS(
        y, conc[["exposure_total", "exposure_conc"]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", clusters=clusters)
    results["+Concentration"] = res

    wins = base.copy()
    lo, hi = wins["exposure_total"].quantile([0.01, 0.99])
    wins["exposure_total_w"] = wins["exposure_total"].clip(lo, hi)
    y = wins["dln_gross_output"]
    clusters = make_clusters(wins)
    res = PanelOLS(y, wins[["exposure_total_w"]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", clusters=clusters
    )
    results["Winsorized 1/99"] = res

    logx = base.copy()
    logx["ln1p_exposure_total"] = np.log1p(logx["exposure_total"])
    y = logx["dln_gross_output"]
    clusters = make_clusters(logx)
    res = PanelOLS(
        y, logx[["ln1p_exposure_total"]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", clusters=clusters)
    results["Log(1+x)"] = res

    lag = df.copy()
    lag["exposure_total_l1"] = lag.groupby(level=0)["exposure_total"].shift(1)
    lag = lag.dropna(subset=["dln_gross_output", "exposure_total_l1"])
    y = lag["dln_gross_output"]
    clusters = make_clusters(lag)
    res = PanelOLS(
        y, lag[["exposure_total_l1"]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", clusters=clusters)
    results["Lagged exposure (t-1)"] = res

    va = df.dropna(subset=["dln_value_added", "exposure_total"])
    y = va["dln_value_added"]
    clusters = make_clusters(va)
    res = PanelOLS(y, va[["exposure_total"]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", clusters=clusters
    )
    results["Value added growth"] = res

    write_regression_table(results, output_csv, output_tex)
    logger.info("Wrote robustness regression table")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_robustness",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )
