"""Table rendering utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import read_parquet, write_csv


_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    for needle, repl in _LATEX_REPLACEMENTS.items():
        text = text.replace(needle, repl)
    return text


def _format_cell(value: object, float_format: str) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, bool)):
        return str(int(value))
    if isinstance(value, float):
        return float_format % value
    return _escape_latex(value)


def _dataframe_to_simple_latex(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    float_format: str = "%.4f",
    index: bool = False,
) -> str:
    frame = df.copy()
    if index:
        frame = frame.reset_index()

    cols = [str(c) for c in frame.columns]
    align = "l" * len(cols)

    lines: list[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    lines.append(rf"\label{{{_escape_latex(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(_escape_latex(c) for c in cols) + r" \\")
    lines.append(r"\hline")
    for _, row in frame.iterrows():
        cells = [_format_cell(row[c], float_format) for c in frame.columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def write_regression_table(results: dict, csv_path: Path, tex_path: Path) -> None:
    rows = []
    cov_diagnostics = []
    for name, res in results.items():
        params = res.params
        # Compute standard errors from the covariance matrix manually so we can:
        # - avoid RuntimeWarning spam when the covariance is not PSD
        # - preserve information (negative/NaN variances become NaN std_err)
        cov = res.cov
        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
        # Symmetrize for numerical stability.
        cov_mat = 0.5 * (cov_mat + cov_mat.T)
        diag = np.diag(cov_mat)
        var_names = list(getattr(cov, "index", params.index))
        se_vals = np.full_like(diag, fill_value=np.nan, dtype=float)
        ok = np.isfinite(diag) & (diag >= 0)
        se_vals[ok] = np.sqrt(diag[ok])
        ses = pd.Series(se_vals, index=var_names, name="std_error")

        # Covariance diagnostics (helps debug non-PSD clustered covariances).
        tol = 1e-12
        n_params = int(cov_mat.shape[0]) if cov_mat.ndim == 2 else 0
        if n_params:
            try:
                eig_min = float(np.min(np.linalg.eigvalsh(cov_mat)))
            except Exception:
                eig_min = float("nan")
            diag_min = float(np.nanmin(diag)) if diag.size else float("nan")
            n_neg_diag = int(np.sum(np.isfinite(diag) & (diag < -tol)))
        else:
            eig_min = float("nan")
            diag_min = float("nan")
            n_neg_diag = 0
        cov_diagnostics.append(
            {
                "model": name,
                "cov_type": getattr(res, "cov_type", ""),
                "n_obs": getattr(res, "nobs", ""),
                "n_params": n_params,
                "cov_min_eigenvalue": eig_min,
                "cov_min_diag": diag_min,
                "cov_negative_diag_count": n_neg_diag,
            }
        )
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

    diag_path = csv_path.with_name(f"{csv_path.stem}_covdiag.csv")
    pd.DataFrame(cov_diagnostics).to_csv(diag_path, index=False)

    tex_path.write_text(
        _dataframe_to_simple_latex(
            table,
            caption="Regression results",
            label="tab:reg",
            float_format="%.4f",
            index=False,
        ),
        encoding="utf-8",
    )


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
    output_tex.write_text(
        _dataframe_to_simple_latex(
            summary,
            caption="Summary statistics",
            label="tab:summary",
            float_format="%.4f",
            index=True,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote summary stats table")
    record_manifest(
        paths,
        config.model_dump(),
        "render_tables",
        [paths.data_final / "panel_annual.parquet"],
        [output_csv, output_tex],
    )
