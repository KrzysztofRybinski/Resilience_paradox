"""Pipeline orchestration and manifest writing."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from resilience_paradox.config import load_config
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.utils.hashing import dict_hash


def _git_hash(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def record_manifest(
    paths: Paths,
    config_payload: dict,
    step: str,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
) -> None:
    manifest_path = paths.output / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in inputs],
        "outputs": [str(path) for path in outputs],
    }
    config_hash = dict_hash(config_payload)
    payload = {
        "git_commit": _git_hash(paths.root),
        "config_hash": config_hash,
        "runs": [record],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing.setdefault("runs", []).append(record)
        existing["git_commit"] = payload["git_commit"]
        existing["config_hash"] = payload["config_hash"]
        manifest_path.write_text(json.dumps(existing, indent=2))
    else:
        manifest_path.write_text(json.dumps(payload, indent=2))


def run_all_pipeline(config_path: str, force: bool = False, sample: bool = False) -> None:
    from resilience_paradox.data.oecd_download import download_icio_bundles
    from resilience_paradox.data.oecd_icio_panels import build_icio_panels
    from resilience_paradox.data.stateaid_clean import clean_stateaid
    from resilience_paradox.data.stateaid_download import download_stateaid
    from resilience_paradox.data.stateaid_panel import build_upstream_aid_panel
    from resilience_paradox.data.exposure import build_exposure, build_panel
    from resilience_paradox.models.fe import estimate_main, estimate_shock, estimate_robustness
    from resilience_paradox.models.localprojections import estimate_event_study
    from resilience_paradox.models.tables import render_all_tables
    from resilience_paradox.models.figures import render_all_figures

    logger = setup_logging()
    config = load_config(config_path)
    paths = Paths.from_config(config)
    paths.ensure()
    payload = config.model_dump()

    logger.info("Starting run-all pipeline")

    download_stateaid(config, force=force, sample=sample)
    record_manifest(paths, payload, "stateaid_download", [], [])

    clean_stateaid(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "stateaid_clean",
        [paths.data_raw / "stateaid"],
        [paths.data_int / "stateaid_awards.parquet"],
    )

    if sample:
        logger.info("Sample mode: skipping OECD ICIO bundle downloads")
        oecd_download_outputs: list[Path] = []
    else:
        download_icio_bundles(config, refresh=False)
        oecd_download_outputs = list(paths.oecd_raw_dir().glob("*.zip"))
    record_manifest(paths, payload, "oecd_download", [], oecd_download_outputs)

    build_icio_panels(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "oecd_build_icio_panels",
        [paths.data_raw / "oecd_icio"],
        [
            paths.data_int / "icio_output_va.parquet",
            paths.data_int / "icio_input_shares_base.parquet",
            paths.data_int / "icio_import_hhi.parquet",
            paths.data_int / "icio_split_weights.parquet",
        ],
    )

    build_upstream_aid_panel(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "stateaid_build_panel",
        [paths.data_int / "stateaid_awards.parquet"],
        [paths.data_final / "upstream_aid_panel.parquet"],
    )

    build_exposure(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "build_exposure",
        [paths.data_final / "upstream_aid_panel.parquet"],
        [paths.data_final / "exposure_panel.parquet"],
    )

    build_panel(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "build_panel",
        [paths.data_final / "exposure_panel.parquet"],
        [paths.data_final / "panel_annual.parquet"],
    )

    if sample:
        logger.info("Sample mode: skipping estimation steps")
    else:
        estimate_main(config, force=force, sample=sample)
        estimate_shock(config, force=force, sample=sample)
        estimate_robustness(config, force=force, sample=sample)
        estimate_event_study(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "estimate",
        [paths.data_final / "panel_annual.parquet"],
        [paths.output / "tables", paths.output / "figures"],
    )

    render_all_tables(config, force=force, sample=sample)
    render_all_figures(config, force=force, sample=sample)
    record_manifest(
        paths,
        payload,
        "render",
        [paths.output / "tables"],
        [paths.output / "figures"],
    )

    logger.info("Pipeline completed")
