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
        return out_path

    raw_path = _raw_prices_dir(paths) / "ecb_eurofxref_hist.csv"
    if not raw_path.exists() or force:
        logger.info("Downloading ECB FX rates: %s", ECB_EUROFXREF_HIST_URL)
        download_file(ECB_EUROFXREF_HIST_URL, raw_path)

    df = pd.read_csv(raw_path)
    if "Date" not in df.columns:
        raise RuntimeError(f"Unexpected ECB FX CSV format at {raw_path} (missing 'Date' column).")
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year.astype(int)
    df = df[df["year"].isin(list(years))].copy()

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

    raw_path = _raw_prices_dir(paths) / "eurostat_prc_hicp_aind.tsv.gz"

    def _download_hicp() -> None:
        logger.info("Downloading Eurostat HICP annual index: %s", EUROSTAT_HICP_AIND_URL)
        download_file(EUROSTAT_HICP_AIND_URL, raw_path)

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
        download_file(url, raw_path)

    if not raw_path.exists() or force:
        try:
            _download_hicp()
        except Exception as exc:
            logger.warning("Primary HICP download failed (%s). Trying legacy URL.", exc)
            try:
                download_file(EUROSTAT_HICP_AIND_URL_LEGACY, raw_path)
            except Exception as exc2:
                logger.warning("Legacy HICP download failed (%s). Trying inventory lookup.", exc2)
                try:
                    _download_from_inventory()
                except Exception as exc3:
                    if raw_path.exists():
                        logger.warning(
                            "Using existing cached HICP file at %s after download failures.", raw_path
                        )
                    else:
                        raise RuntimeError(
                            "Failed to download Eurostat HICP annual index. "
                            "Please download prc_hicp_aind.tsv.gz manually and place it at "
                            f"{raw_path}."
                        ) from exc3

    raw = pd.read_csv(raw_path, sep="\t", compression="gzip", dtype=str)
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
