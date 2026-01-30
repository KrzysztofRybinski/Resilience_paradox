"""Local projections / event study estimation."""
from __future__ import annotations

import numpy as np
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
    diag_csv = paths.output / "tables" / "eventstudy_input_diagnostics.csv"
    if output_csv.exists() and not force:
        logger.info("Event study coefficients already exist; skipping.")
        return
    if force:
        output_csv.unlink(missing_ok=True)
        output_csv.with_suffix(".parquet").unlink(missing_ok=True)

    df = read_parquet(paths.data_final / "panel_annual.parquet")
    df = df[df["year"].between(config.years.start, config.years.end)]
    df = df.sort_values(["country_iso3", "icio50", "year"])
    df["entity_id"] = df["country_iso3"].astype(str) + "_" + df["icio50"].astype(str)
    df["interaction"] = df["exposure_total"] * df["post_covid"]

    # Diagnostics: is the interaction ever non-zero in the post period?
    diag = (
        df.assign(
            has_exposure=df["exposure_total"].notna(),
            interaction_nonzero=df["interaction"].fillna(0) != 0,
        )
        .groupby("year", as_index=False)
        .agg(
            n_obs=("year", "size"),
            share_has_exposure=("has_exposure", "mean"),
            share_interaction_nonzero=("interaction_nonzero", "mean"),
        )
        .sort_values("year")
    )
    diag_csv.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(diag_csv, index=False)

    post_years = df.loc[df["post_covid"].fillna(False), "year"]
    if post_years.empty:
        raise RuntimeError(
            "post_covid is never true in the panel; cannot run event study. "
            f"See {diag_csv}."
        )
    min_post_year = int(post_years.min())
    max_year = int(df["year"].max())
    max_horizon_feasible = max_year - min_post_year

    if diag.loc[diag["year"] >= min_post_year, "share_interaction_nonzero"].fillna(0).sum() == 0:
        raise RuntimeError(
            "Event-study regressor (exposure_total × post_covid) has no non-zero values "
            "in post-covid years after filtering. "
            "This usually means exposure_total is missing/zero in 2020+ due to upstream mapping "
            "or incomplete State Aid coverage. "
            f"See {diag_csv} for year-by-year coverage."
        )

    rows = []
    for horizon in range(4):
        if horizon > max_horizon_feasible:
            rows.append(
                {
                    "horizon": horizon,
                    "coef": pd.NA,
                    "se": pd.NA,
                    "n_obs": 0,
                    "reason": "horizon_not_observable_in_sample",
                }
            )
            continue
        df["y_h"] = df.groupby(["entity_id"])["dln_gross_output"].shift(-horizon)
        sample_df = df.dropna(subset=["y_h", "interaction"]).set_index(["entity_id", "year"]).sort_index()
        if sample_df.empty:
            rows.append({"horizon": horizon, "coef": pd.NA, "se": pd.NA, "n_obs": 0, "reason": "empty"})
            continue
        if sample_df["interaction"].nunique() <= 1:
            rows.append(
                {
                    "horizon": horizon,
                    "coef": pd.NA,
                    "se": pd.NA,
                    "n_obs": int(sample_df.shape[0]),
                    "reason": "no_regressor_variation",
                }
            )
            continue
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
            drop_absorbed=True,
        )
        res = model.fit(cov_type="clustered", clusters=clusters)
        coef = res.params.get("interaction", pd.NA)
        # Avoid non-PSD covariance sqrt warnings; treat negative variances as missing.
        cov = res.cov
        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov, dtype=float)
        cov_mat = 0.5 * (cov_mat + cov_mat.T)
        var = float(np.diag(cov_mat)[0]) if cov_mat.size else float("nan")
        se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else pd.NA
        rows.append(
            {
                "horizon": horizon,
                "coef": coef,
                "se": se,
                "n_obs": int(res.nobs),
                "reason": "" if pd.notna(se) else "non_psd_covariance_or_missing_se",
            }
        )

    out = pd.DataFrame(rows)
    write_parquet(out, output_csv.with_suffix(".parquet"))
    out.to_csv(output_csv, index=False)
    logger.info("Wrote event study coefficients")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_event_study",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, diag_csv],
    )
