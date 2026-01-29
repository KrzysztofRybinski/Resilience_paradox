"""Local projections / event study estimation."""
from __future__ import annotations

import pandas as pd
from linearmodels.panel import PanelOLS

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_parquet


def estimate_event_study(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_csv = paths.output / "tables" / "eventstudy_coeffs.csv"
    if output_csv.exists() and not force:
        logger.info("Event study coefficients already exist; skipping.")
        return

    df = read_parquet(paths.data_final / "panel_annual.parquet")
    df = df[df["year"].between(config.years.start, config.years.end)]
    df = df.sort_values(["country_iso3", "icio50", "year"])
    df["entity_id"] = df["country_iso3"].astype(str) + "_" + df["icio50"].astype(str)
    df["interaction"] = df["exposure_total"] * df["post_covid"]

    rows = []
    for horizon in range(4):
        df["y_h"] = df.groupby(["entity_id"])["dln_gross_output"].shift(-horizon)
        sample_df = df.dropna(subset=["y_h", "interaction"]).set_index(["entity_id", "year"]).sort_index()
        y = sample_df["y_h"]
        clusters = pd.DataFrame(
            {
                "country_iso3": sample_df["country_iso3"].astype(str),
                "year": sample_df.index.get_level_values("year"),
            },
            index=sample_df.index,
        )
        model = PanelOLS(
            y,
            sample_df[["interaction"]],
            entity_effects=True,
            time_effects=True,
        )
        res = model.fit(cov_type="clustered", clusters=clusters)
        coef = res.params.get("interaction", pd.NA)
        se = res.std_errors.get("interaction", pd.NA)
        rows.append({"horizon": horizon, "coef": coef, "se": se})

    out = pd.DataFrame(rows)
    write_parquet(out, output_csv.with_suffix(".parquet"))
    out.to_csv(output_csv, index=False)
    logger.info("Wrote event study coefficients")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_event_study",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv],
    )
