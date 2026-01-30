"""Figure rendering utilities."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_csv


def _save_hist(series: pd.Series, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna(), bins=30, color="steelblue", alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_log_hist(
    series: pd.Series,
    path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 30,
    base: float = 10.0,
) -> None:
    """Histogram for heavy-tailed positive series using log-transform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = series.dropna()
    zeros = int((values == 0).sum())
    pos = values[values > 0]
    if pos.empty:
        plt.figure(figsize=(6, 4))
        plt.title(f"{title} (no positive values)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return

    log_vals = np.log(pos.astype(float)) / math.log(base)
    plt.figure(figsize=(7, 4))
    plt.hist(log_vals, bins=bins, color="steelblue", alpha=0.7)
    plt.title(f"{title} (log{int(base)}; zeros={zeros:,})")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _sample_per_group(
    df: pd.DataFrame, group_col: str, *, n: int, random_state: int = 0
) -> pd.DataFrame:
    """Deterministically sample up to `n` rows per group.

    Avoids GroupBy.apply because newer pandas versions can drop the grouping column.
    """
    take = int(n)
    if take <= 0:
        return df.head(0).copy()
    rng = np.random.RandomState(int(random_state))
    sampled = df.copy()
    sampled["_rand"] = rng.rand(sampled.shape[0])
    sampled = sampled.sort_values([group_col, "_rand"])
    sampled = sampled.groupby(group_col, sort=False, as_index=False).head(take)
    return sampled.drop(columns=["_rand"])


def _save_aid_amounts_local_vs_eur_by_country(
    awards: pd.DataFrame, path: Path, *, per_country: int = 2500
) -> None:
    """Grid comparing distributions of aid in local currency vs EUR (log scale)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needed = {"country_iso3", "aid_amount_original", "aid_amount_eur"}
    missing = needed - set(awards.columns)
    if missing:
        raise RuntimeError(f"State aid awards missing columns for diagnostics: {sorted(missing)}")

    df = awards[["country_iso3", "aid_amount_original", "aid_amount_eur", "aid_currency"]].copy()
    df = df.dropna(subset=["country_iso3", "aid_amount_original", "aid_amount_eur"])
    df = df[(df["aid_amount_original"] > 0) & (df["aid_amount_eur"] > 0)]
    if df.empty:
        raise RuntimeError("No positive aid amounts available for diagnostics plotting.")

    countries = sorted(df["country_iso3"].dropna().astype(str).unique().tolist())
    ncols = 5
    nrows = int(math.ceil(len(countries) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 3.0 * nrows), sharey=False)
    axes_arr = np.asarray(axes).reshape(-1)

    sampled = _sample_per_group(df, "country_iso3", n=per_country)
    sampled["log10_original"] = np.log10(sampled["aid_amount_original"].astype(float))
    sampled["log10_eur"] = np.log10(sampled["aid_amount_eur"].astype(float))

    for idx, country in enumerate(countries):
        ax = axes_arr[idx]
        subset = sampled[sampled["country_iso3"].astype(str) == country]
        if subset.empty:
            ax.axis("off")
            continue
        ax.hist(
            subset["log10_original"].dropna(),
            bins=25,
            histtype="step",
            linewidth=1.0,
            color="tab:orange",
            label="local",
        )
        ax.hist(
            subset["log10_eur"].dropna(),
            bins=25,
            histtype="step",
            linewidth=1.0,
            color="tab:blue",
            label="EUR",
        )
        non_eur_share = float((subset["aid_currency"].astype(str).str.upper() != "EUR").mean())
        ax.set_title(f"{country} (non-EUR={non_eur_share:.0%})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_arr[len(countries) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("State aid award amounts: local currency vs EUR (log10 scale)", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(path)
    plt.close(fig)


def _save_naive_vs_converted_country_year_totals(
    awards: pd.DataFrame, path: Path
) -> pd.DataFrame:
    """Compare country-year totals under naive vs converted EUR interpretation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = awards[["country_iso3", "year", "aid_amount_original", "aid_amount_eur", "aid_currency"]].copy()
    df = df.dropna(subset=["country_iso3", "year", "aid_amount_original", "aid_amount_eur"])
    df = df[(df["aid_amount_original"] > 0) & (df["aid_amount_eur"] > 0)]
    if df.empty:
        raise RuntimeError("No positive aid amounts available for country-year totals diagnostics.")

    totals = (
        df.groupby(["country_iso3", "year"], as_index=False)
        .agg(
            naive_eur_million=("aid_amount_original", lambda s: float(s.sum()) / 1e6),
            converted_eur_million=("aid_amount_eur", lambda s: float(s.sum()) / 1e6),
            n_awards=("aid_amount_eur", "size"),
            share_non_eur=("aid_currency", lambda s: float((s.astype(str).str.upper() != "EUR").mean())),
        )
        .copy()
    )
    totals = totals[(totals["naive_eur_million"] > 0) & (totals["converted_eur_million"] > 0)].copy()
    totals["naive_over_converted"] = totals["naive_eur_million"] / totals["converted_eur_million"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    x = totals["converted_eur_million"]
    y = totals["naive_eur_million"]
    axes[0].scatter(x, y, s=14, alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    xmin = max(float(x.min()), 1e-9)
    xmax = max(float(x.max()), xmin * 1.01)
    axes[0].plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", linewidth=1)
    axes[0].set_title("Country-year totals: naive vs converted")
    axes[0].set_xlabel("Converted total (EUR mn, log)")
    axes[0].set_ylabel("Naive total (treat local as EUR, EUR mn, log)")

    ratio = totals["naive_over_converted"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(np.log10(ratio), bins=30, color="slateblue", alpha=0.7)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("log10(naive/converted) across country-years")
    axes[1].set_xlabel("log10 ratio (0 = equal)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return totals


def _save_exposure_coverage_by_year(exposure: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "year" not in exposure.columns or "exposure_total" not in exposure.columns:
        raise RuntimeError("Exposure panel missing required columns for diagnostics.")

    df = exposure[["year", "exposure_total", "post_covid"]].copy()
    df["has_exposure"] = df["exposure_total"].notna()
    df["is_zero"] = df["exposure_total"].fillna(0) == 0
    df["is_positive"] = df["exposure_total"].fillna(0) > 0

    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            n_obs=("exposure_total", "size"),
            share_non_missing=("has_exposure", "mean"),
            share_zero=("is_zero", "mean"),
            share_positive=("is_positive", "mean"),
            mean_exposure=("exposure_total", "mean"),
            p50_exposure=(
                "exposure_total",
                lambda s: float(s.dropna().median()) if s.dropna().shape[0] else np.nan,
            ),
        )
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(yearly["year"], yearly["share_positive"], marker="o", label="share > 0")
    ax.plot(yearly["year"], yearly["share_zero"], marker="o", label="share = 0 (incl missing as 0)")
    ax.set_ylim(0, 1)
    ax.set_title("Exposure_total coverage by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of rows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return yearly


def render_all_figures(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    aid_panel = read_parquet(paths.data_final / "upstream_aid_panel.parquet")
    exposure = read_parquet(paths.data_final / "exposure_panel.parquet")

    fig1 = paths.output / "figures" / "Figure1_aid_concentration_distribution.png"
    fig2 = paths.output / "figures" / "Figure2_exposure_distribution.png"
    fig3 = paths.output / "figures" / "Figure3_eventstudy_output.png"
    fig_s1 = paths.output / "figures" / "FigureS1_stateaid_local_vs_eur_by_country.png"
    fig_s2 = paths.output / "figures" / "FigureS2_stateaid_naive_vs_converted_totals.png"
    fig_s3 = paths.output / "figures" / "FigureS3_exposure_coverage_by_year.png"

    diag_totals_csv = paths.output / "tables" / "stateaid_country_year_totals_naive_vs_converted.csv"
    diag_exposure_csv = paths.output / "tables" / "exposure_total_coverage_by_year.csv"

    if not fig1.exists() or force:
        _save_hist(aid_panel["hhi_beneficiary"], fig1, "Aid concentration (HHI)")

    if not fig2.exists() or force:
        _save_log_hist(
            exposure["exposure_total"],
            fig2,
            title="Exposure total distribution",
            xlabel="log10(exposure_total) for positive values",
            bins=35,
        )

    coeffs_path = paths.output / "tables" / "eventstudy_coeffs.csv"
    if coeffs_path.exists() and (not fig3.exists() or force):
        coeffs = pd.read_csv(coeffs_path)
        if (
            {"horizon", "coef", "se"}.issubset(coeffs.columns)
            and coeffs[["coef", "se"]].fillna(0).to_numpy().sum() != 0
        ):
            plt.figure(figsize=(6, 4))
            plt.plot(coeffs["horizon"], coeffs["coef"], marker="o")
            mask = coeffs["se"].notna() & coeffs["coef"].notna()
            if mask.any():
                x = coeffs.loc[mask, "horizon"]
                y = coeffs.loc[mask, "coef"]
                se = coeffs.loc[mask, "se"]
                plt.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.2)
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.title("Event study: exposure x post-covid")
            plt.xlabel("Horizon")
            plt.ylabel("Coefficient")
            plt.tight_layout()
            plt.savefig(fig3)
            plt.close()
        else:
            logger.warning(
                "Event study coefficients at %s appear degenerate (all zeros/NA). "
                "Run `rp estimate event-study --force` to regenerate before rendering Figure 3.",
                coeffs_path,
            )

    if not sample:
        awards_path = paths.data_int / "stateaid_awards.parquet"
        if awards_path.exists():
            awards = read_parquet(awards_path)
            if not fig_s1.exists() or force:
                _save_aid_amounts_local_vs_eur_by_country(awards, fig_s1)
            if not fig_s2.exists() or force or not diag_totals_csv.exists():
                totals = _save_naive_vs_converted_country_year_totals(awards, fig_s2)
                write_csv(totals, diag_totals_csv)
            if not fig_s3.exists() or force or not diag_exposure_csv.exists():
                yearly = _save_exposure_coverage_by_year(exposure, fig_s3)
                write_csv(yearly, diag_exposure_csv)

    logger.info("Rendered figures")
    record_manifest(
        paths,
        config.model_dump(),
        "render_figures",
        [paths.data_final / "upstream_aid_panel.parquet", paths.data_int / "stateaid_awards.parquet"],
        [fig1, fig2, fig3, fig_s1, fig_s2, fig_s3, diag_totals_csv, diag_exposure_csv],
    )
