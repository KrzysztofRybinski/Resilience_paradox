"""Table rendering utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_csv


def write_regression_table(results: dict, csv_path: Path, tex_path: Path) -> None:
    rows = []
    for name, res in results.items():
        params = res.params
        ses = res.std_errors
        for param_name, value in params.items():
            rows.append(
                {
                    "model": name,
                    "variable": param_name,
                    "coef": value,
                    "std_err": ses.get(param_name, pd.NA),
                    "n_obs": res.nobs,
                }
            )
    table = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    tex_path.write_text(table.to_latex(index=False, float_format="%.4f", caption="Regression results", label="tab:reg"))


def render_all_tables(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_csv = paths.output / "tables" / "Table1_summary_stats.csv"
    output_tex = paths.output / "tables" / "Table1_summary_stats.tex"
    if output_csv.exists() and output_tex.exists() and not force:
        logger.info("Summary stats already exist; skipping.")
        return

    df = read_parquet(paths.data_final / "panel_annual.parquet")
    summary = df[["gross_output", "value_added", "exposure_total", "exposure_conc", "exposure_aidlevel"]].describe().T
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv)
    output_tex.write_text(summary.to_latex(float_format="%.4f", caption="Summary statistics", label="tab:summary"))
    logger.info("Wrote summary stats table")
    record_manifest(
        paths,
        config.model_dump(),
        "render_tables",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )
