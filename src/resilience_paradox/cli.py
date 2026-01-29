"""CLI entrypoint for Resilience Paradox."""
from __future__ import annotations

import typer

from resilience_paradox.config import load_config
from resilience_paradox.data.oecd_download import download_icio_bundles
from resilience_paradox.data.oecd_icio_panels import build_icio_panels
from resilience_paradox.data.stateaid_db import build_stateaid_db
from resilience_paradox.data.stateaid_clean import clean_stateaid
from resilience_paradox.data.stateaid_download import download_stateaid, request_stateaid_email_exports
from resilience_paradox.data.stateaid_panel import build_upstream_aid_panel
from resilience_paradox.data.exposure import build_exposure as build_exposure_panel
from resilience_paradox.data.exposure import build_panel as build_final_panel
from resilience_paradox.models.fe import estimate_main as run_estimate_main
from resilience_paradox.models.fe import estimate_shock as run_estimate_shock
from resilience_paradox.models.fe import estimate_robustness as run_estimate_robustness
from resilience_paradox.models.localprojections import estimate_event_study
from resilience_paradox.models.tables import render_all_tables
from resilience_paradox.models.figures import render_all_figures
from resilience_paradox.pipeline import run_all_pipeline

app = typer.Typer(help="Resilience Paradox pipeline")


@app.command()
def run_all(
    config: str = typer.Option(..., "--config", help="Path to config TOML"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    """Run the full pipeline."""
    run_all_pipeline(config, force=force, sample=sample)


stateaid_app = typer.Typer(help="State aid pipeline")

oecd_app = typer.Typer(help="OECD ICIO pipeline")

build_app = typer.Typer(help="Build derived datasets")

estimate_app = typer.Typer(help="Estimate econometric models")

render_app = typer.Typer(help="Render tables/figures")

prices_app = typer.Typer(help="FX/deflator data")


@stateaid_app.command("download")
def stateaid_download(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    download_stateaid(cfg, force=force, sample=sample)


@stateaid_app.command("build-db")
def stateaid_build_db(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force", help="Rebuild the DuckDB tables from scratch."),
) -> None:
    """Ingest portal CSV exports into DuckDB and print summary stats."""
    cfg = load_config(config)
    build_stateaid_db(cfg, force=force)


@stateaid_app.command("request-email")
def stateaid_request_email(
    config: str = typer.Option(..., "--config"),
    first_name: str = typer.Option(..., "--first-name"),
    last_name: str = typer.Option(..., "--last-name"),
    email: str = typer.Option(..., "--email"),
    select_all_countries: bool = typer.Option(True, "--select-all/--select-config-countries"),
    force: bool = typer.Option(False, "--force"),
    headed: bool = typer.Option(False, "--headed", help="Run with a visible browser window."),
    year: list[int] = typer.Option([], "--year", help="Limit to specific year(s); repeatable."),
) -> None:
    """Submit export-by-email requests on the State Aid portal (does not download files)."""
    cfg = load_config(config)
    years = year if year else None
    request_stateaid_email_exports(
        cfg,
        first_name=first_name,
        last_name=last_name,
        email=email,
        force=force,
        select_all_countries=select_all_countries,
        years=years,
        headless=None if not headed else False,
    )


@stateaid_app.command("clean")
def stateaid_clean(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    clean_stateaid(cfg, force=force, sample=sample)


@stateaid_app.command("from-csv")
def stateaid_from_csv(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    """Build the State Aid DB from portal CSV exports and clean it to Parquet."""
    cfg = load_config(config)
    build_stateaid_db(cfg, force=force)
    clean_stateaid(cfg, force=force, sample=sample)


@stateaid_app.command("build-panel")
def stateaid_build_panel(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    build_upstream_aid_panel(cfg, force=force, sample=sample)


@oecd_app.command("download")
def oecd_download(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    cfg = load_config(config)
    download_icio_bundles(cfg, refresh=force)


@oecd_app.command("build-icio-panels")
def oecd_build_icio_panels(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    build_icio_panels(cfg, force=force, sample=sample)


@build_app.command("exposure")
def build_exposure(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    build_exposure_panel(cfg, force=force, sample=sample)


@build_app.command("panel")
def build_panel(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    build_final_panel(cfg, force=force, sample=sample)


@estimate_app.command("main")
def estimate_main(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    run_estimate_main(cfg, force=force, sample=sample)


@estimate_app.command("shock")
def estimate_shock(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    run_estimate_shock(cfg, force=force, sample=sample)


@estimate_app.command("robustness")
def estimate_robustness(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    run_estimate_robustness(cfg, force=force, sample=sample)


@prices_app.command("download")
def prices_download(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Download FX rates (ECB) and deflators (Eurostat) used for currency conversion/deflation."""
    from resilience_paradox.data.prices import ensure_ecb_fx_annual, ensure_hicp_deflator
    from resilience_paradox.paths import Paths

    cfg = load_config(config)
    paths = Paths.from_config(cfg)
    paths.ensure()
    ensure_ecb_fx_annual(paths, cfg, force=force)
    ensure_hicp_deflator(paths, cfg, force=force)


@render_app.command("all")
def render_all(
    config: str = typer.Option(..., "--config"),
    force: bool = typer.Option(False, "--force"),
    sample: bool = typer.Option(False, "--sample"),
) -> None:
    cfg = load_config(config)
    render_all_tables(cfg, force=force, sample=sample)
    render_all_figures(cfg, force=force, sample=sample)


app.add_typer(stateaid_app, name="stateaid")
app.add_typer(oecd_app, name="oecd")
app.add_typer(build_app, name="build")
app.add_typer(estimate_app, name="estimate")
app.add_typer(render_app, name="render")
app.add_typer(prices_app, name="prices")


if __name__ == "__main__":
    app()
