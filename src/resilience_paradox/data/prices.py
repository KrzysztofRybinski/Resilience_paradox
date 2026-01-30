"""Price and exchange-rate helpers (FX + deflators).

We use:
- ECB eurofxref daily exchange rates (1 EUR = X currency units) to build annual-average FX rates.
- Eurostat HICP annual average index to build a EUR deflator (e.g., real 2020 EUR).

All downloads are cached under:
- data/raw/prices/ (raw source files)
- data/intermediate/prices/ (processed Parquet tables used by the pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.utils.http import download_file
from resilience_paradox.utils.io import read_parquet, write_parquet

ECB_EUROFXREF_HIST_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
ECB_EUROFXREF_HIST_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ECB_DATA_API_BASE = "https://data-api.ecb.europa.eu/service/data"

# Some legacy currencies can appear in State Aid exports even after euro adoption due to
# data-entry mistakes. For these, use the irrevocably fixed conversion rates.
# Rates are expressed as: 1 EUR = X currency units (i.e., "rate_per_eur").
FIXED_EUR_CONVERSION_RATES: dict[str, float] = {
    # Lithuania adopted the euro on 2015-01-01. Official fixed rate: 1 EUR = 3.45280 LTL.
    "LTL": 3.45280,
}

# Eurostat migration note: BulkDownloadListing endpoints are being phased out.
# Prefer the new API (files?file=...), and fall back to inventory lookup if needed.
EUROSTAT_FILES_API = "https://ec.europa.eu/eurostat/api/dissemination/files"
EUROSTAT_INVENTORY_URL = "https://ec.europa.eu/eurostat/api/dissemination/files/inventory?type=data"
EUROSTAT_HICP_RELATIVE_PATH = "data/prc_hicp_aind.tsv.gz"
EUROSTAT_HICP_AIND_URL = f"{EUROSTAT_FILES_API}?file={EUROSTAT_HICP_RELATIVE_PATH}"
EUROSTAT_HICP_AIND_URL_LEGACY = (
    "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/"
    "BulkDownloadListing?downfile=data/prc_hicp_aind.tsv.gz"
)


@dataclass(frozen=True)
class PriceConfig:
    base_year: int = 2020
    hicp_geo: str = "EA19"


def _get_price_config(config: AppConfig) -> PriceConfig:
    payload = config.model_dump()
    cfg = payload.get("prices") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    base_year = int(cfg.get("base_year", 2020))
    hicp_geo = str(cfg.get("hicp_geo", "EA19"))
    return PriceConfig(base_year=base_year, hicp_geo=hicp_geo)


def _raw_prices_dir(paths: Paths) -> Path:
    path = paths.data_raw / "prices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _int_prices_dir(paths: Paths) -> Path:
    path = paths.data_int / "prices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_fixed_eur_rates(
    annual: pd.DataFrame, years: range, logger
) -> pd.DataFrame:
    if not FIXED_EUR_CONVERSION_RATES:
        return annual
    annual = annual.copy()
    annual["currency"] = annual["currency"].astype(str).str.strip().str.upper()
    annual["year"] = annual["year"].astype(int)

    for currency, rate_per_eur in FIXED_EUR_CONVERSION_RATES.items():
        existing = set(annual.loc[annual["currency"] == currency, "year"].tolist())
        missing = [year for year in years if year not in existing]
        if not missing:
            continue
        logger.warning(
            "Applying fixed EUR conversion rate for %s (rate_per_eur=%s) for missing years: %s",
            currency,
            rate_per_eur,
            missing,
        )
        annual = pd.concat(
            [
                annual,
                pd.DataFrame(
                    {
                        "year": missing,
                        "currency": currency,
                        "rate_per_eur": float(rate_per_eur),
                    }
                ),
            ],
            ignore_index=True,
        )

    annual = annual.drop_duplicates(subset=["year", "currency"], keep="last")
    return annual


def ensure_ecb_fx_annual(paths: Paths, config: AppConfig, *, force: bool = False) -> Path:
    """Ensure annual-average ECB FX rates exist and return the Parquet path.

    Output schema:
      - year (int)
      - currency (str, ISO 3-letter)
      - rate_per_eur (float): currency units per 1 EUR
    """

    logger = setup_logging()
    years = range(int(config.years.base_start), int(config.years.end) + 1)
    out_path = _int_prices_dir(paths) / "fx_ecb_annual.parquet"
    if out_path.exists() and not force:
        # Validate cache to avoid subtle failures when the upstream CSV download is truncated,
        # or when we add fixed-rate backfills (e.g., LTL) after the parquet was generated.
        try:
            cached = read_parquet(out_path)
            cached["currency"] = cached["currency"].astype(str).str.strip().str.upper()
            cached_years_usd = set(
                cached.loc[cached["currency"] == "USD", "year"].astype(int).tolist()
            )
            missing_usd = [y for y in years if y not in cached_years_usd]

            missing_fixed: dict[str, list[int]] = {}
            for ccy in FIXED_EUR_CONVERSION_RATES:
                cached_years_ccy = set(
                    cached.loc[cached["currency"] == ccy, "year"].astype(int).tolist()
                )
                missing = [y for y in years if y not in cached_years_ccy]
                if missing:
                    missing_fixed[ccy] = missing

            if not missing_usd and not missing_fixed:
                return out_path

            logger.warning(
                "Cached FX table at %s is incomplete (missing USD years=%s; missing fixed-rate years=%s). Rebuilding.",
                out_path,
                missing_usd,
                missing_fixed,
            )
        except Exception as exc:
            logger.warning(
                "Could not validate cached FX table at %s (%s). Rebuilding.", out_path, exc
            )

    raw_path = _raw_prices_dir(paths) / "ecb_eurofxref_hist.csv"
    if not raw_path.exists() or force:
        logger.info("Downloading ECB FX rates: %s", ECB_EUROFXREF_HIST_URL)
        download_file(ECB_EUROFXREF_HIST_URL, raw_path)

    def _read_fx_csv(path: Path) -> pd.DataFrame:
        df_local = pd.read_csv(path)
        df_local.columns = [str(c).strip() for c in df_local.columns]
        if "Date" not in df_local.columns:
            raise RuntimeError(
                f"Unexpected ECB FX CSV format at {path} (missing 'Date' column)."
            )
        df_local = df_local.rename(columns={"Date": "date"})
        df_local["date"] = pd.to_datetime(df_local["date"], errors="coerce")
        df_local = df_local.dropna(subset=["date"]).copy()
        df_local["year"] = df_local["date"].dt.year.astype(int)
        df_local = df_local[df_local["year"].isin(list(years))].copy()
        return df_local

    def _missing_years(current: pd.DataFrame) -> list[int]:
        present = set(current["year"].dropna().astype(int).unique())
        return sorted([y for y in years if y not in present])

    df = _read_fx_csv(raw_path)
    missing_years = _missing_years(df)
    if missing_years:
        logger.warning(
            "ECB FX CSV at %s is missing years %s. Trying ZIP fallback.",
            raw_path,
            missing_years,
        )
        zip_path = _raw_prices_dir(paths) / "ecb_eurofxref_hist.zip"
        try:
            download_file(ECB_EUROFXREF_HIST_ZIP_URL, zip_path)
            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith("eurofxref-hist.csv"):
                        zf.extract(name, raw_path.parent)
                        extracted = raw_path.parent / name
                        extracted.replace(raw_path)
                        break
            df = _read_fx_csv(raw_path)
            missing_years = _missing_years(df)
        except Exception as exc:
            logger.warning("ZIP fallback failed (%s).", exc)

    if missing_years:
        logger.warning(
            "ECB eurofxref CSV still missing years %s. Falling back to ECB Data API.",
            missing_years,
        )

        api_path = _raw_prices_dir(paths) / "ecb_exr_monthly.csv"
        start_period = f"{min(years)}-01"
        end_period = f"{max(years)}-12"
        api_url = (
            f"{ECB_DATA_API_BASE}/EXR/M..EUR.SP00.A"
            f"?format=csvdata&startPeriod={start_period}&endPeriod={end_period}"
        )
        try:
            download_file(api_url, api_path)
            api_df = pd.read_csv(api_path, dtype=str)
        except Exception as exc:
            raise RuntimeError(
                "ECB FX data does not cover all required years and ECB Data API fallback failed. "
                f"Missing: {missing_years}. "
                f"Please download the latest eurofxref-hist.csv from {ECB_EUROFXREF_HIST_URL} "
                f"and place it at {raw_path}."
            ) from exc

        api_df.columns = [str(c).strip().upper() for c in api_df.columns]
        time_col = "TIME_PERIOD" if "TIME_PERIOD" in api_df.columns else None
        value_col = "OBS_VALUE" if "OBS_VALUE" in api_df.columns else None
        currency_col = "CURRENCY" if "CURRENCY" in api_df.columns else None
        denom_col = "CURRENCY_DENOM" if "CURRENCY_DENOM" in api_df.columns else None
        if not (time_col and value_col and currency_col):
            raise RuntimeError(
                f"Unexpected ECB Data API CSV format at {api_path} (missing columns)."
            )
        if denom_col:
            api_df = api_df[api_df[denom_col].astype(str).str.upper() == "EUR"]

        api_df["year"] = api_df[time_col].astype(str).str.slice(0, 4).astype(int)
        api_df["rate_per_eur"] = pd.to_numeric(api_df[value_col], errors="coerce")
        api_df["currency"] = api_df[currency_col].astype(str).str.strip().str.upper()
        api_df = api_df.dropna(subset=["year", "rate_per_eur", "currency"])
        api_df = api_df[api_df["year"].isin(list(years))].copy()

        annual = (
            api_df.groupby(["year", "currency"], as_index=False)["rate_per_eur"]
            .mean()
            .sort_values(["year", "currency"])
        )
        eur = pd.DataFrame({"year": list(years), "currency": "EUR", "rate_per_eur": 1.0})
        annual = pd.concat([annual, eur], ignore_index=True).drop_duplicates(
            subset=["year", "currency"], keep="last"
        )
        annual = _apply_fixed_eur_rates(annual, years, logger)
        write_parquet(annual, out_path)
        logger.info("Wrote annual FX rates (ECB API fallback) to %s", out_path)
        return out_path

    long = df.melt(id_vars=["year"], var_name="currency", value_name="rate_per_eur")
    long["currency"] = long["currency"].astype(str).str.strip().str.upper()
    long["rate_per_eur"] = pd.to_numeric(long["rate_per_eur"], errors="coerce")
    long = long.dropna(subset=["currency", "rate_per_eur"])

    annual = (
        long.groupby(["year", "currency"], as_index=False)["rate_per_eur"]
        .mean()
        .sort_values(["year", "currency"])
    )

    # Ensure EUR exists so downstream joins always succeed.
    eur = pd.DataFrame({"year": list(years), "currency": "EUR", "rate_per_eur": 1.0})
    annual = pd.concat([annual, eur], ignore_index=True).drop_duplicates(
        subset=["year", "currency"], keep="last"
    )
    annual = _apply_fixed_eur_rates(annual, years, logger)

    write_parquet(annual, out_path)
    logger.info("Wrote annual FX rates to %s", out_path)
    return out_path


def _parse_eurostat_value(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text == ":":
        return None
    # Eurostat TSVs sometimes include flags after the numeric value ("100.0 p").
    numeric = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    try:
        return float(numeric)
    except ValueError:
        return None


def ensure_hicp_deflator(paths: Paths, config: AppConfig, *, force: bool = False) -> Path:
    """Ensure an annual EUR deflator exists and return the Parquet path.

    Uses Eurostat HICP annual average index (2015=100), geo configurable (default EA19).

    Output schema:
      - year (int)
      - hicp_index (float)
      - deflator_to_base (float): multiply nominal EUR by this to get real EUR in base_year prices
    """

    logger = setup_logging()
    price_cfg = _get_price_config(config)
    years = range(int(config.years.base_start), int(config.years.end) + 1)
    out_path = _int_prices_dir(paths) / f"hicp_{price_cfg.hicp_geo}_annual.parquet"
    if out_path.exists() and not force:
        return out_path

    expected_path = _raw_prices_dir(paths) / "eurostat_prc_hicp_aind.tsv.gz"

    def _find_existing_hicp() -> Path | None:
        if expected_path.exists():
            return expected_path
        candidates = sorted(_raw_prices_dir(paths).glob("*prc_hicp_aind*.tsv.gz"))
        return candidates[0] if candidates else None

    existing_path = _find_existing_hicp()
    raw_path = existing_path or expected_path

    def _download_hicp() -> None:
        logger.info("Downloading Eurostat HICP annual index: %s", EUROSTAT_HICP_AIND_URL)
        download_file(EUROSTAT_HICP_AIND_URL, expected_path)

    def _download_from_inventory() -> None:
        inventory_path = _raw_prices_dir(paths) / "eurostat_inventory_data.txt"
        logger.info("Fetching Eurostat inventory listing: %s", EUROSTAT_INVENTORY_URL)
        download_file(EUROSTAT_INVENTORY_URL, inventory_path)
        rel_path = None
        with inventory_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "prc_hicp_aind.tsv.gz" not in line:
                    continue
                # Try to extract a plausible relative path token.
                tokens = [tok.strip() for tok in line.replace("|", " ").split()]
                for tok in tokens:
                    if tok.endswith("prc_hicp_aind.tsv.gz"):
                        rel_path = tok
                        break
                if rel_path:
                    break
        if not rel_path:
            raise RuntimeError(
                "Could not locate prc_hicp_aind.tsv.gz in Eurostat inventory listing."
            )
        url = f"{EUROSTAT_FILES_API}?file={rel_path}"
        logger.info("Downloading HICP via inventory path: %s", url)
        download_file(url, expected_path)

    if force or existing_path is None:
        try:
            _download_hicp()
            raw_path = expected_path
        except Exception as exc:
            logger.warning("Primary HICP download failed (%s). Trying legacy URL.", exc)
            try:
                download_file(EUROSTAT_HICP_AIND_URL_LEGACY, expected_path)
                raw_path = expected_path
            except Exception as exc2:
                logger.warning("Legacy HICP download failed (%s). Trying inventory lookup.", exc2)
                try:
                    _download_from_inventory()
                    raw_path = expected_path
                except Exception as exc3:
                    if existing_path and existing_path.exists():
                        logger.warning(
                            "Using existing cached HICP file at %s after download failures.",
                            existing_path,
                        )
                        raw_path = existing_path
                    else:
                        raise RuntimeError(
                            "Failed to download Eurostat HICP annual index. "
                            "Please download prc_hicp_aind.tsv.gz manually and place it at "
                            f"{expected_path}."
                        ) from exc3

    raw = pd.read_csv(raw_path, sep="\t", compression="gzip", dtype=str)
    raw.columns = [str(col).strip() for col in raw.columns]
    first_col = raw.columns[0]
    dims_part = first_col.split("\\")[0]
    dims = [d.strip() for d in dims_part.split(",") if d.strip()]
    if not dims:
        raise RuntimeError(f"Unexpected Eurostat TSV format at {raw_path} (no dimensions).")

    key = raw[first_col].astype(str)
    split = key.str.split(",", expand=True)
    if split.shape[1] != len(dims):
        raise RuntimeError(
            f"Unexpected Eurostat TSV format at {raw_path} (dims {dims} do not match key)."
        )
    for idx, dim in enumerate(dims):
        raw[dim] = split[idx].astype(str).str.strip()
    raw = raw.drop(columns=[first_col])

    # HICP annual average index, all items (overall inflation).
    if "unit" not in raw.columns or "coicop" not in raw.columns:
        raise RuntimeError(
            f"Unexpected Eurostat TSV format at {raw_path} (missing 'unit'/'coicop')."
        )

    geo_col = "geo" if "geo" in raw.columns else None
    if geo_col is None:
        raise RuntimeError(f"Unexpected Eurostat TSV format at {raw_path} (missing 'geo').")

    subset = raw[
        (raw["unit"] == "INX_A_AVG")
        & (raw["coicop"] == "CP00")
        & (raw[geo_col] == price_cfg.hicp_geo)
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"No HICP series found for unit=INX_A_AVG, coicop=CP00, geo={price_cfg.hicp_geo} in {raw_path}."
        )

    year_cols = [c for c in subset.columns if c.isdigit()]
    if not year_cols:
        raise RuntimeError(f"Unexpected Eurostat TSV format at {raw_path} (no year columns).")

    long_rows: list[dict] = []
    for col in year_cols:
        year = int(col)
        if year not in years:
            continue
        value = _parse_eurostat_value(subset.iloc[0][col])
        if value is None:
            continue
        long_rows.append({"year": year, "hicp_index": float(value)})

    df = pd.DataFrame(long_rows).sort_values("year")
    if df.empty:
        raise RuntimeError(
            f"HICP series for {price_cfg.hicp_geo} contained no usable values for {min(years)}-{max(years)}."
        )

    base_row = df[df["year"] == price_cfg.base_year]
    if base_row.empty:
        raise RuntimeError(
            f"Missing HICP value for base_year={price_cfg.base_year} (geo={price_cfg.hicp_geo}). "
            f"Available years: {sorted(df['year'].tolist())}"
        )
    base_index = float(base_row["hicp_index"].iloc[0])
    df["deflator_to_base"] = base_index / df["hicp_index"]

    write_parquet(df, out_path)
    logger.info("Wrote HICP deflator to %s", out_path)
    return out_path


def load_fx_annual(paths: Paths, config: AppConfig) -> pd.DataFrame:
    path = ensure_ecb_fx_annual(paths, config, force=False)
    return read_parquet(path)


def load_hicp_deflator(paths: Paths, config: AppConfig) -> pd.DataFrame:
    path = ensure_hicp_deflator(paths, config, force=False)
    return read_parquet(path)
