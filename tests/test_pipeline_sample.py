from resilience_paradox.config import load_config
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import run_all_pipeline


def test_pipeline_sample(monkeypatch, tmp_path):
    config = load_config("config/default.toml")
    paths = Paths.from_config(config, root=tmp_path)
    paths.ensure()

    def mock_download_icio(*args, **kwargs):
        return None

    def mock_estimate(*args, **kwargs):
        tables_dir = paths.output / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / "Table2_baseline_effects.csv").write_text("model,coef\n")
        (tables_dir / "Table2_baseline_effects.tex").write_text("%")
        (tables_dir / "Table3_shock_interactions.csv").write_text("model,coef\n")
        (tables_dir / "Table3_shock_interactions.tex").write_text("%")
        (tables_dir / "eventstudy_coeffs.csv").write_text("horizon,coef,se\n0,0,0\n")

    def mock_render(*args, **kwargs):
        figures_dir = paths.output / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / "Figure1_aid_concentration_distribution.png").write_text("fake")
        (figures_dir / "Figure2_exposure_distribution.png").write_text("fake")
        (figures_dir / "Figure3_eventstudy_output.png").write_text("fake")
        tables_dir = paths.output / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / "Table1_summary_stats.csv").write_text("col,mean\n")
        (tables_dir / "Table1_summary_stats.tex").write_text("%")

    monkeypatch.setattr("resilience_paradox.pipeline.Paths.from_config", lambda cfg: paths)
    monkeypatch.setattr(
        "resilience_paradox.data.oecd_download.download_icio_bundles", mock_download_icio
    )
    monkeypatch.setattr("resilience_paradox.models.fe.estimate_main", mock_estimate)
    monkeypatch.setattr("resilience_paradox.models.fe.estimate_shock", mock_estimate)
    monkeypatch.setattr("resilience_paradox.models.localprojections.estimate_event_study", mock_estimate)
    monkeypatch.setattr("resilience_paradox.models.tables.render_all_tables", mock_render)
    monkeypatch.setattr("resilience_paradox.models.figures.render_all_figures", mock_render)

    run_all_pipeline("config/default.toml", force=True, sample=True)

    assert (paths.data_final / "panel_annual.parquet").exists()
    assert (paths.data_final / "exposure_panel.parquet").exists()
    assert (paths.data_final / "upstream_aid_panel.parquet").exists()
    assert (paths.output / "run_manifest.json").exists()
