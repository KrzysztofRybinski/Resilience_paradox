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


def _select_fe_flags(df: pd.DataFrame, columns: list[str], logger) -> tuple[bool, bool]:
    """Choose FE flags that keep regressors identified.

    If a regressor has no cross-sectional variation in any year, time effects absorb it.
    If a regressor has no within-entity variation, entity effects absorb it.
    """

    entity_effects = True
    time_effects = True
    for col in columns:
        series = df[col].dropna()
        if series.nunique() <= 1:
            zeros = int((series == 0).sum())
            raise RuntimeError(
                f"Regressor {col} has no variation after filtering; cannot estimate. "
                f"(n={series.shape[0]}, unique={series.nunique()}, zeros={zeros}, "
                f"min={series.min()}, max={series.max()}). "
                "This usually indicates a data issue upstream (e.g., missing FX/HICP conversion "
                "or aid totals collapsing to 0). Check reports in "
                "data/intermediate/stateaid_checks/ and rerun `rp prices download` and "
                "`rp stateaid clean`."
            )
        within_entity = series.groupby(level=0).var()
        within_year = series.groupby(level=1).var()
        if (within_entity.fillna(0) == 0).all():
            entity_effects = False
        if (within_year.fillna(0) == 0).all():
            time_effects = False

    if not entity_effects and not time_effects:
        raise RuntimeError(
            "Regressors are fully absorbed by fixed effects; "
            "no identified variation remains."
        )

    if not entity_effects or not time_effects:
        logger.warning(
            "Adjusting fixed effects for identification (entity_effects=%s, time_effects=%s).",
            entity_effects,
            time_effects,
        )

    return entity_effects, time_effects


def _ensure_full_rank(exog: pd.DataFrame, logger) -> pd.DataFrame:
    """Drop collinear columns to ensure full column rank."""

    if exog.shape[1] <= 1:
        return exog
    selected: list[str] = []
    for col in exog.columns:
        candidate = exog[selected + [col]]
        if np.linalg.matrix_rank(candidate.to_numpy()) > len(selected):
            selected.append(col)
        else:
            logger.warning("Dropping collinear regressor: %s", col)
    if not selected:
        raise RuntimeError("All regressors are collinear; cannot estimate.")
    return exog[selected]


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

    exog1 = _ensure_full_rank(df[["exposure_total"]], logger)
    entity_effects, time_effects = _select_fe_flags(df, list(exog1.columns), logger)
    model1 = PanelOLS(
        y,
        exog1,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    )
    res1 = model1.fit(cov_type="clustered", clusters=clusters)

    exog2 = _ensure_full_rank(df[["exposure_total", "exposure_conc"]], logger)
    entity_effects, time_effects = _select_fe_flags(df, list(exog2.columns), logger)
    model2 = PanelOLS(
        y,
        exog2,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
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

    exog = _ensure_full_rank(df[["exposure_covid", "exposure_energy"]], logger)
    entity_effects, time_effects = _select_fe_flags(df, list(exog.columns), logger)
    model = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
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
    exog = _ensure_full_rank(base[["exposure_total"]], logger)
    entity_effects, time_effects = _select_fe_flags(base, list(exog.columns), logger)
    res = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)
    results["Baseline"] = res

    no_2016 = base[base.index.get_level_values("year") != 2016]
    if not no_2016.empty:
        y = no_2016["dln_gross_output"]
        clusters = make_clusters(no_2016)
        exog = _ensure_full_rank(no_2016[["exposure_total"]], logger)
        entity_effects, time_effects = _select_fe_flags(no_2016, list(exog.columns), logger)
        res = PanelOLS(
            y,
            exog,
            entity_effects=entity_effects,
            time_effects=time_effects,
            drop_absorbed=True,
        ).fit(cov_type="clustered", clusters=clusters)
        results["Drop 2016"] = res
        # Two-way clustered covariance can be non-PSD with few time clusters; if it yields
        # a negative variance, also report a one-way country-clustered SE for reference.
        try:
            cov_mat = res.cov.to_numpy()
            cov_mat = 0.5 * (cov_mat + cov_mat.T)
            var = float(np.diag(cov_mat)[0]) if cov_mat.size else float("nan")
        except Exception:
            var = float("nan")
        if not np.isfinite(var) or var < 0:
            logger.warning(
                "Non-PSD clustered covariance detected for 'Drop 2016' (var=%s). "
                "Also fitting one-way country clustering for a stable SE.",
                var,
            )
            res_country = PanelOLS(
                y,
                exog,
                entity_effects=entity_effects,
                time_effects=time_effects,
                drop_absorbed=True,
            ).fit(cov_type="clustered", clusters=no_2016[["country_iso3"]])
            results["Drop 2016 (cluster country)"] = res_country

    conc = df.dropna(subset=["dln_gross_output", "exposure_total", "exposure_conc"])
    y = conc["dln_gross_output"]
    clusters = make_clusters(conc)
    exog = _ensure_full_rank(conc[["exposure_total", "exposure_conc"]], logger)
    entity_effects, time_effects = _select_fe_flags(conc, list(exog.columns), logger)
    res = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)
    results["+Concentration"] = res

    wins = base.copy()
    lo, hi = wins["exposure_total"].quantile([0.01, 0.99])
    wins["exposure_total_w"] = wins["exposure_total"].clip(lo, hi)
    y = wins["dln_gross_output"]
    clusters = make_clusters(wins)
    exog = _ensure_full_rank(wins[["exposure_total_w"]], logger)
    entity_effects, time_effects = _select_fe_flags(wins, list(exog.columns), logger)
    res = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)
    results["Winsorized 1/99"] = res

    logx = base.copy()
    logx["ln1p_exposure_total"] = np.log1p(logx["exposure_total"])
    y = logx["dln_gross_output"]
    clusters = make_clusters(logx)
    exog = _ensure_full_rank(logx[["ln1p_exposure_total"]], logger)
    entity_effects, time_effects = _select_fe_flags(logx, list(exog.columns), logger)
    res = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)
    results["Log(1+x)"] = res

    lag = df.copy()
    lag["exposure_total_l1"] = lag.groupby(level=0)["exposure_total"].shift(1)
    lag = lag.dropna(subset=["dln_gross_output", "exposure_total_l1"])
    y = lag["dln_gross_output"]
    clusters = make_clusters(lag)
    exog = _ensure_full_rank(lag[["exposure_total_l1"]], logger)
    entity_effects, time_effects = _select_fe_flags(lag, list(exog.columns), logger)
    res = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
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


def estimate_currency_sanity(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    """Sanity check: replicate the *wrong* "treat local currency as EUR" scaling.

    This is *not* used in the research pipeline; it exists to explain sign changes when
    moving from unconverted local-currency amounts to proper EUR conversion.
    """

    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    output_csv = paths.output / "tables" / "TableS_currency_sanity.csv"
    output_tex = paths.output / "tables" / "TableS_currency_sanity.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Currency sanity table already exists; skipping.")
        return

    # 1) Baseline "correct" exposure_total regression (uses converted EUR + deflator pipeline).
    df = read_parquet(paths.data_final / "panel_annual.parquet")
    df = df[df["year"].between(config.years.start, config.years.end)]
    df = df.dropna(subset=["dln_gross_output", "exposure_total"])

    df["entity_id"] = df["country_iso3"].astype(str) + "_" + df["icio50"].astype(str)
    df = df.set_index(["entity_id", "year"]).sort_index()

    y = df["dln_gross_output"]
    clusters = pd.DataFrame(
        {
            "country_iso3": df["country_iso3"].astype(str),
            "year": df.index.get_level_values("year"),
        },
        index=df.index,
    )
    exog = _ensure_full_rank(df[["exposure_total"]], logger)
    entity_effects, time_effects = _select_fe_flags(df, list(exog.columns), logger)
    res_correct = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)

    # 2) Build a "naive" exposure_total that treats local-currency aid amounts as if they were EUR.
    awards = read_parquet(paths.data_int / "stateaid_awards.parquet")
    needed = {"country_iso3", "year", "aid_amount_original", "aid_amount_eur"}
    missing = needed - set(awards.columns)
    if missing:
        raise RuntimeError(
            "State aid awards missing required columns for currency sanity check: "
            f"{sorted(missing)}"
        )

    tmp = awards.dropna(subset=["country_iso3", "year", "aid_amount_original", "aid_amount_eur"]).copy()
    tmp = tmp[(tmp["aid_amount_original"] > 0) & (tmp["aid_amount_eur"] > 0)]
    if tmp.empty:
        raise RuntimeError("No usable state aid award amounts found for currency sanity check.")

    ratio = (
        tmp.groupby(["country_iso3", "year"], as_index=False)
        .agg(
            naive_sum=("aid_amount_original", "sum"),
            converted_sum=("aid_amount_eur", "sum"),
        )
        .copy()
    )
    ratio["naive_over_converted"] = ratio["naive_sum"] / ratio["converted_sum"]
    ratio = ratio[["country_iso3", "year", "naive_over_converted"]]

    upstream = read_parquet(paths.data_final / "upstream_aid_panel.parquet")
    if "aid_total_eur_million" not in upstream.columns:
        raise RuntimeError("Upstream aid panel missing aid_total_eur_million (cannot build naive exposure).")

    upstream = upstream.merge(ratio, on=["country_iso3", "year"], how="left")
    # EUR countries should be ~1; default to 1 if no ratio row exists.
    upstream["naive_over_converted"] = upstream["naive_over_converted"].fillna(1.0)
    upstream["aid_total_naive_eur_million"] = upstream["aid_total_eur_million"] * upstream["naive_over_converted"]

    output_va = read_parquet(paths.data_int / "icio_output_va.parquet")
    if sample:
        output_va["usd_per_eur"] = 1.0
    else:
        from resilience_paradox.data.prices import load_fx_annual

        fx = load_fx_annual(paths, config)
        usd = fx[fx["currency"] == "USD"][["year", "rate_per_eur"]].rename(
            columns={"rate_per_eur": "usd_per_eur"}
        )
        output_va = output_va.merge(usd, on="year", how="left")
        if output_va["usd_per_eur"].isna().any():
            missing_years = sorted(output_va.loc[output_va["usd_per_eur"].isna(), "year"].unique().tolist())
            raise RuntimeError(
                "Missing USD/EUR FX rates for some ICIO years: "
                f"{missing_years}. Run `rp prices download --force`."
            )

    output_va["gross_output_eur_million"] = output_va["gross_output"] / output_va["usd_per_eur"]

    merged = upstream.merge(
        output_va[["country_iso3", "icio50", "year", "gross_output_eur_million"]],
        on=["country_iso3", "icio50", "year"],
        how="left",
    )
    merged["subsidy_intensity_naive"] = merged["aid_total_naive_eur_million"] / merged["gross_output_eur_million"]
    merged = merged.rename(columns={"icio50": "upstream_icio50"})

    shares = read_parquet(paths.data_int / "icio_input_shares_base.parquet")
    exposure = shares.merge(merged[["country_iso3", "upstream_icio50", "year", "subsidy_intensity_naive"]], on=["country_iso3", "upstream_icio50"], how="left")
    exposure["exposure_total_naive"] = exposure["ioshare_base"] * exposure["subsidy_intensity_naive"]

    def _sum_or_na(values: pd.Series) -> float:
        return float(values.sum(min_count=1))  # type: ignore[arg-type]

    naive = (
        exposure.groupby(["country_iso3", "downstream_icio50", "year"], as_index=False)
        .agg(exposure_total_naive=("exposure_total_naive", _sum_or_na))
        .rename(columns={"downstream_icio50": "icio50"})
    )

    # Merge into the panel and estimate with the same FE/clustering.
    panel_raw = read_parquet(paths.data_final / "panel_annual.parquet")
    panel_raw = panel_raw[panel_raw["year"].between(config.years.start, config.years.end)]
    panel_raw = panel_raw.dropna(subset=["dln_gross_output"])
    panel_raw = panel_raw.merge(naive, on=["country_iso3", "icio50", "year"], how="left")
    panel_raw = panel_raw.dropna(subset=["exposure_total_naive"])
    panel_raw["entity_id"] = panel_raw["country_iso3"].astype(str) + "_" + panel_raw["icio50"].astype(str)
    panel_raw = panel_raw.set_index(["entity_id", "year"]).sort_index()

    y = panel_raw["dln_gross_output"]
    clusters = pd.DataFrame(
        {
            "country_iso3": panel_raw["country_iso3"].astype(str),
            "year": panel_raw.index.get_level_values("year"),
        },
        index=panel_raw.index,
    )
    exog = _ensure_full_rank(panel_raw[["exposure_total_naive"]], logger)
    entity_effects, time_effects = _select_fe_flags(panel_raw, list(exog.columns), logger)
    res_naive = PanelOLS(
        y,
        exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters)

    write_regression_table(
        {"Correct (converted EUR)": res_correct, "Naive (treat local as EUR)": res_naive},
        output_csv,
        output_tex,
    )
    logger.info("Wrote currency sanity regression table")
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_currency_sanity",
        [
            paths.data_final / "panel_annual.parquet",
            paths.data_final / "upstream_aid_panel.parquet",
            paths.data_int / "stateaid_awards.parquet",
        ],
        [output_csv, output_tex],
    )


def estimate_country_effects(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    """Estimate baseline slopes separately by country.

    This is useful to explore heterogeneity in the main hypotheses (H1/H2) across countries.
    We run within-country sector panels with sector and year fixed effects.
    """

    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    output_csv = paths.output / "tables" / "TableS_country_effects.csv"
    output_tex = paths.output / "tables" / "TableS_country_effects.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Country effects table already exists; skipping.")
        return

    from resilience_paradox.models.tables import _dataframe_to_simple_latex

    panel = read_parquet(paths.data_final / "panel_annual.parquet")
    panel = panel[panel["year"].between(config.years.start, config.years.end)]
    panel = panel.dropna(subset=["dln_gross_output"])

    def _safe_se(res, var: str) -> float:
        try:
            cov = res.cov
            cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
            cov_mat = 0.5 * (cov_mat + cov_mat.T)
            names = list(getattr(cov, "index", res.params.index))
            if var not in names:
                return float("nan")
            diag = np.diag(cov_mat)
            idx = names.index(var)
            v = float(diag[idx])
            if not np.isfinite(v) or v < 0:
                return float("nan")
            return float(np.sqrt(v))
        except Exception:
            return float("nan")

    rows: list[dict[str, object]] = []
    countries = sorted(panel["country_iso3"].dropna().astype(str).unique().tolist())
    for country in countries:
        subset = panel[panel["country_iso3"].astype(str) == country].copy()
        subset["entity_id"] = subset["icio50"].astype(str)
        subset = subset.set_index(["entity_id", "year"]).sort_index()

        clusters = pd.DataFrame(
            {"entity_id": subset.index.get_level_values("entity_id")},
            index=subset.index,
        )

        out: dict[str, object] = {
            "country_iso3": country,
            "n_obs_total_only": 0,
            "coef_total_only": np.nan,
            "se_total_only": np.nan,
            "p_total_only": np.nan,
            "n_obs_concentration": 0,
            "coef_total_ctrl": np.nan,
            "se_total_ctrl": np.nan,
            "p_total_ctrl": np.nan,
            "coef_concentration": np.nan,
            "se_concentration": np.nan,
            "p_concentration": np.nan,
            "error_total_only": "",
            "error_concentration": "",
        }

        total_df = subset.dropna(subset=["exposure_total", "dln_gross_output"])
        out["n_obs_total_only"] = int(total_df.shape[0])
        if total_df.shape[0]:
            try:
                y = total_df["dln_gross_output"]
                exog = _ensure_full_rank(total_df[["exposure_total"]], logger)
                entity_effects, time_effects = _select_fe_flags(total_df, list(exog.columns), logger)
                res = PanelOLS(
                    y,
                    exog,
                    entity_effects=entity_effects,
                    time_effects=time_effects,
                    drop_absorbed=True,
                ).fit(cov_type="clustered", clusters=clusters.loc[total_df.index])
                out["coef_total_only"] = float(res.params.get("exposure_total", np.nan))
                out["se_total_only"] = _safe_se(res, "exposure_total")
                out["p_total_only"] = float(getattr(res, "pvalues", {}).get("exposure_total", np.nan))
            except Exception as exc:
                out["error_total_only"] = str(exc)

        conc_df = subset.dropna(subset=["exposure_total", "exposure_conc", "dln_gross_output"])
        out["n_obs_concentration"] = int(conc_df.shape[0])
        if conc_df.shape[0]:
            try:
                y = conc_df["dln_gross_output"]
                exog = _ensure_full_rank(conc_df[["exposure_total", "exposure_conc"]], logger)
                entity_effects, time_effects = _select_fe_flags(conc_df, list(exog.columns), logger)
                res = PanelOLS(
                    y,
                    exog,
                    entity_effects=entity_effects,
                    time_effects=time_effects,
                    drop_absorbed=True,
                ).fit(cov_type="clustered", clusters=clusters.loc[conc_df.index])
                out["coef_total_ctrl"] = float(res.params.get("exposure_total", np.nan))
                out["se_total_ctrl"] = _safe_se(res, "exposure_total")
                out["p_total_ctrl"] = float(getattr(res, "pvalues", {}).get("exposure_total", np.nan))
                out["coef_concentration"] = float(res.params.get("exposure_conc", np.nan))
                out["se_concentration"] = _safe_se(res, "exposure_conc")
                out["p_concentration"] = float(getattr(res, "pvalues", {}).get("exposure_conc", np.nan))
            except Exception as exc:
                out["error_concentration"] = str(exc)

        rows.append(out)

    results = pd.DataFrame(rows).sort_values("country_iso3")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    output_tex.write_text(
        _dataframe_to_simple_latex(
            results,
            caption="Country-specific slopes (baseline and concentration specifications)",
            label="tab:country_effects",
            float_format="%.4f",
            index=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote country effects table to %s", output_csv)
    record_manifest(
        paths,
        config.model_dump(),
        "estimate_country_effects",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )
