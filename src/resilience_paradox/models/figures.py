"""Figure rendering utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet


def _save_hist(series: pd.Series, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(series.dropna(), bins=30, color="steelblue", alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_all_figures(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    aid_panel = read_parquet(paths.data_final / "upstream_aid_panel.parquet")
    exposure = read_parquet(paths.data_final / "exposure_panel.parquet")

    fig1 = paths.output / "figures" / "Figure1_aid_concentration_distribution.png"
    fig2 = paths.output / "figures" / "Figure2_exposure_distribution.png"
    fig3 = paths.output / "figures" / "Figure3_eventstudy_output.png"

    if not fig1.exists() or force:
        _save_hist(aid_panel["hhi_beneficiary"], fig1, "Aid concentration (HHI)")

    if not fig2.exists() or force:
        _save_hist(exposure["exposure_total"], fig2, "Exposure total distribution")

    coeffs_path = paths.output / "tables" / "eventstudy_coeffs.csv"
    if coeffs_path.exists() and (not fig3.exists() or force):
        coeffs = pd.read_csv(coeffs_path)
        plt.figure(figsize=(6, 4))
        plt.plot(coeffs["horizon"], coeffs["coef"], marker="o")
        plt.fill_between(
            coeffs["horizon"],
            coeffs["coef"] - 1.96 * coeffs["se"],
            coeffs["coef"] + 1.96 * coeffs["se"],
            alpha=0.2,
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.title("Event study: exposure x post-covid")
        plt.xlabel("Horizon")
        plt.ylabel("Coefficient")
        plt.tight_layout()
        plt.savefig(fig3)
        plt.close()

    logger.info("Rendered figures")
    record_manifest(
        paths,
        config.model_dump(),
        "render_figures",
        [paths.data_final / "upstream_aid_panel.parquet"],
        [fig1, fig2, fig3],
    )
