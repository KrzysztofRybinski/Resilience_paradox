# Resilience Paradox (EU-only)

Pipeline to build EU state-aid exposure panels and regression outputs based on OECD ICIO data.

## Quickstart

```bash
git clone git@github.com:KrzysztofRybinski/Resilience_paradox.git
cd Resilience_paradox
uv sync
uv run rp --help
uv run rp run-all --config config/default.toml
```

### Playwright setup (state aid downloads)

If you use the Playwright backend for the State Aid Transparency portal:

```bash
uv run playwright install chromium
```

## Outputs

After a successful run you should find:

- `data/final/panel_annual.parquet`
- `data/final/upstream_aid_panel.parquet`
- `data/final/exposure_panel.parquet`
- `output/tables/Table1_summary_stats.tex` and `.csv`
- `output/tables/Table2_baseline_effects.tex` and `.csv`
- `output/tables/Table3_shock_interactions.tex` and `.csv`
- `output/figures/Figure1_aid_concentration_distribution.png`
- `output/figures/Figure2_exposure_distribution.png`
- `output/figures/Figure3_eventstudy_output.png`
- `output/run_manifest.json`

## Sample mode (smoke test)

Run a fast sample mode (2 countries × 2 years) that finishes in minutes:

```bash
uv run rp run-all --config config/default.toml --sample
```

## Step-by-step pipeline

You can run steps independently:

```bash
uv run rp stateaid download --config config/default.toml
uv run rp stateaid clean --config config/default.toml
uv run rp stateaid build-panel --config config/default.toml
uv run rp oecd download --config config/default.toml
uv run rp oecd build-icio-panels --config config/default.toml
uv run rp build exposure --config config/default.toml
uv run rp build panel --config config/default.toml
uv run rp estimate main --config config/default.toml
uv run rp estimate shock --config config/default.toml
uv run rp render all --config config/default.toml
```

## Notes on external sources

- OECD ICIO bundles are downloadable from the OECD ICIO dataset page.
- The State Aid Transparency portal lists Poland, Romania, Spain, and Slovenia as using national registers; these are excluded in the EU-only baseline.
- Optional fallback: the EUSA database (MIT-licensed) provides state aid awards for 2016–2020.
