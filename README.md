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

### Manual portal CSV exports (no Playwright)

If you download CSV exports manually (or receive them by email), place them anywhere under `data/raw/stateaid/`
(you can use subfolders; any folder starting with `_` is ignored), then run:

```bash
uv run rp stateaid from-csv --config config/default.toml
```

`rp stateaid build-db` also writes coverage checks to `data/intermediate/stateaid_checks/` to help catch
missing/overlapping manual downloads (e.g., global time-horizon gaps and duplicate `Ref-no.`).

Note: `rp run-all` still runs the automated State Aid download step; for manual exports, prefer the step-by-step commands.

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

Run a fast sample mode (small synthetic dataset) that finishes in minutes:

```bash
uv run rp run-all --config config/default.toml --sample
```

Notes:
- Sample mode skips OECD bundle downloads and generates minimal ICIO/state-aid inputs locally.
- LaTeX outputs are written without the optional `jinja2` dependency.

## Step-by-step pipeline

You can run steps independently:

```bash
uv run rp stateaid download --config config/default.toml
uv run rp stateaid request-email --config config/default.toml --first-name FIRST --last-name LAST --email you@example.com
uv run rp stateaid from-csv --config config/default.toml
uv run rp prices download --config config/default.toml
uv run rp stateaid build-panel --config config/default.toml
uv run rp oecd download --config config/default.toml
uv run rp oecd build-icio-panels --config config/default.toml
uv run rp build exposure --config config/default.toml
uv run rp build panel --config config/default.toml
uv run rp estimate main --config config/default.toml
uv run rp estimate shock --config config/default.toml
uv run rp estimate robustness --config config/default.toml
uv run rp render all --config config/default.toml
```

Notes:
- `rp stateaid from-csv` runs `rp stateaid build-db` + `rp stateaid clean`. `rp stateaid build-db` scans `data/raw/stateaid/**/*.{csv,csv.gz}`, builds `data/intermediate/stateaid.duckdb`, prints basic row-count statistics, and writes coverage checks (missing country-years, duplicated `Ref-no.`, global time-horizon gaps, inferred file segments) to `data/intermediate/stateaid_checks/`. It ignores any directories starting with `_` (like `_debug` and `_email_requests`).
- `rp stateaid request-email` **does not download files**. It submits “export by email” requests on the portal and records progress in `data/raw/stateaid/_email_requests/requests.csv` so you can resume safely.
- Use `--select-all` (default) to select all portal countries on the landing page, or `--select-config-countries` to select only the countries from your config (excluding `exclude_iso3`).
- `config/default.toml` includes the EU23 sample plus the United Kingdom (`config/countries_eu23_plusuk.csv`).

## Notes on external sources

- OECD ICIO bundles are downloadable from the OECD ICIO dataset page.
- The State Aid Transparency portal lists Poland, Romania, Spain, and Slovenia as using national registers; these are excluded in the EU-only baseline.
- Optional fallback: the EUSA database (MIT-licensed) provides state aid awards for 2016–2020.
