"""Build ICIO panels for output/VA, base IO shares, and import HHI."""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import polars as pl
import numpy as np

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import write_parquet
from resilience_paradox.data.oecd_icio_reader import (
    read_icio_long,
    compute_output_va,
    compute_input_shares_base,
    compute_import_hhi,
)


FINAL_DEMAND_CODES = {"HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"}


def _resolve_bundle_zip(raw_dir: Path, release: str, bundle: str) -> Path:
    expected = raw_dir / f"ICIO{release}_{bundle}.zip"
    if expected.exists():
        return expected
    candidates = sorted(raw_dir.glob(f"*{bundle}*.zip"))
    if candidates:
        return candidates[0]
    # Some OECD downloads replace '-' with '_' in filenames.
    candidates = sorted(raw_dir.glob(f"*{bundle.replace('-', '_')}*.zip"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Missing OECD ICIO ZIP for bundle {bundle}. Expected {expected} (or any *{bundle}*.zip) in {raw_dir}."
    )


def _extract_first_csv(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV files found inside {zip_path}")
        target = extract_dir / Path(members[0]).name
        if not target.exists():
            archive.extract(members[0], extract_dir)
            extracted = extract_dir / members[0]
            extracted.rename(target)
    return target


def _zip_members_by_year(zip_path: Path) -> dict[int, str]:
    """Return mapping of year -> member name for SML-style bundles containing per-year CSVs."""
    members: dict[int, str] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            m = re.match(r"^(?P<year>\d{4})", Path(name).name)
            if not m:
                continue
            year = int(m.group("year"))
            members[year] = name
    return members


def _parse_country_code(label: str) -> tuple[str | None, str | None]:
    if "_" not in label:
        return None, None
    country, code = label.split("_", 1)
    country = country.strip()
    code = code.strip()
    if len(country) != 3:
        return None, None
    return country, code


def _is_sector_code(code: str) -> bool:
    return code and code not in FINAL_DEMAND_CODES and code != "OUT"


def _process_sml_year(
    *,
    zip_path: Path,
    member: str,
    year: int,
    dest_countries: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Process a single SML wide ICIO table and return outputs.

    Returns:
      - output_va rows for selected dest industries
      - import hhi rows for selected dest industries
      - sector share matrix (n_sectors x n_dest) for this year
      - sector code list (length n_sectors)
      - dest column labels as tuple-encoded strings "<ISO3>|<SECTOR>" (length n_dest)
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(member, "r") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text)
            header = next(reader)

            # Identify sector columns and countries available in the table.
            sector_codes: set[str] = set()
            origin_countries: set[str] = set()
            for col in header[1:]:
                col = str(col).strip()
                if not col or col == "OUT":
                    continue
                ctry, code = _parse_country_code(col)
                if not ctry or not code or not _is_sector_code(code):
                    continue
                sector_codes.add(code)
                origin_countries.add(ctry)

            sector_list = sorted(sector_codes)
            country_list = sorted(origin_countries)
            sector_to_idx = {s: i for i, s in enumerate(sector_list)}
            country_to_idx = {c: i for i, c in enumerate(country_list)}

            # Pick destination industry columns (restricted to dest_countries and sector codes).
            dest_indices: list[int] = []
            dest_country: list[str] = []
            dest_sector: list[str] = []
            for idx, col in enumerate(header):
                if idx == 0:
                    continue
                col = str(col).strip()
                if not col or col == "OUT":
                    continue
                ctry, code = _parse_country_code(col)
                if not ctry or not code:
                    continue
                if ctry not in dest_countries:
                    continue
                if not _is_sector_code(code):
                    continue
                dest_indices.append(idx)
                dest_country.append(ctry)
                dest_sector.append(code)

            if not dest_indices:
                raise RuntimeError(
                    f"No destination industry columns matched requested countries in {zip_path}:{member}"
                )

            n_dest = len(dest_indices)
            n_countries = len(country_list)
            n_sectors = len(sector_list)

            totals = np.zeros(n_dest, dtype=np.float64)
            by_country = np.zeros((n_countries, n_dest), dtype=np.float64)
            by_sector = np.zeros((n_sectors, n_dest), dtype=np.float64)

            gross_output = np.full(n_dest, np.nan, dtype=np.float64)
            value_added = np.full(n_dest, np.nan, dtype=np.float64)

            for row in reader:
                if not row:
                    continue
                label = str(row[0]).strip()

                if label == "VA":
                    value_added = np.fromiter(
                        (float(row[i] or 0.0) for i in dest_indices),
                        dtype=np.float64,
                        count=n_dest,
                    )
                    continue
                if label == "OUT":
                    gross_output = np.fromiter(
                        (float(row[i] or 0.0) for i in dest_indices),
                        dtype=np.float64,
                        count=n_dest,
                    )
                    continue
                if "_" not in label:
                    # TLS, blanks, etc.
                    continue

                origin, code = _parse_country_code(label)
                if not origin or not code:
                    continue
                if code not in sector_to_idx:
                    continue
                country_idx = country_to_idx.get(origin)
                if country_idx is None:
                    continue

                values = np.fromiter(
                    (float(row[i] or 0.0) for i in dest_indices),
                    dtype=np.float64,
                    count=n_dest,
                )
                totals += values
                by_country[country_idx] += values
                by_sector[sector_to_idx[code]] += values

            # Compute import HHI and import share (intermediate only).
            with np.errstate(divide="ignore", invalid="ignore"):
                shares_country = np.divide(
                    by_country, totals, out=np.zeros_like(by_country), where=totals != 0
                )
                import_hhi = np.sum(shares_country**2, axis=0)

                dest_country_idx = np.array(
                    [country_to_idx.get(c, -1) for c in dest_country], dtype=int
                )
                cols = np.arange(n_dest, dtype=int)
                same_country_inputs = np.where(
                    dest_country_idx >= 0,
                    by_country[dest_country_idx, cols],
                    0.0,
                )
                import_share = np.divide(
                    totals - same_country_inputs,
                    totals,
                    out=np.full_like(totals, np.nan),
                    where=totals != 0,
                )
                import_hhi = np.where(totals != 0, import_hhi, np.nan)

                shares_sector = np.divide(
                    by_sector, totals, out=np.zeros_like(by_sector), where=totals != 0
                )

            output_va = pd.DataFrame(
                {
                    "country_iso3": dest_country,
                    "icio50": dest_sector,
                    "year": year,
                    "gross_output": gross_output,
                    "value_added": value_added,
                }
            )
            hhi = pd.DataFrame(
                {
                    "country_iso3": dest_country,
                    "icio50": dest_sector,
                    "year": year,
                    "import_hhi_intermediate": import_hhi,
                    "import_share_intermediate": import_share,
                }
            )

            dest_labels = [f"{c}|{s}" for c, s in zip(dest_country, dest_sector, strict=False)]
            return output_va, hhi, shares_sector, sector_list, dest_labels


def build_icio_panels(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    output_path = paths.data_int / "icio_output_va.parquet"
    shares_path = paths.data_int / "icio_input_shares_base.parquet"
    hhi_path = paths.data_int / "icio_import_hhi.parquet"

    split_path = paths.data_int / "icio_split_weights.parquet"
    if (
        not force
        and output_path.exists()
        and shares_path.exists()
        and hhi_path.exists()
        and split_path.exists()
    ):
        logger.info("ICIO panels already exist; skipping.")
        return

    if sample:
        logger.info("Building sample ICIO panels")
        df = pl.DataFrame(
            {
                "year": [2013, 2013, 2016, 2016],
                "origin_country": ["FRA", "DEU", "FRA", "DEU"],
                "origin_sector": ["C24A", "C24A", "C24A", "C24A"],
                "dest_country": ["FRA", "FRA", "FRA", "FRA"],
                "dest_sector": ["C24A", "C24A", "C24A", "C24A"],
                "flow_type": ["intermediate", "intermediate", "go", "va"],
                "value": [100.0, 50.0, 500.0, 200.0],
            }
        )
        output_va = compute_output_va(df)
        shares = compute_input_shares_base(df, range(config.years.base_start, config.years.base_end + 1))
        hhi = compute_import_hhi(df)
    else:
        raw_dir = paths.oecd_raw_dir()
        # Resolve bundle ZIPs (handles manual downloads with arbitrary filenames).
        bundle_zips = [
            _resolve_bundle_zip(raw_dir, config.oecd.icio.release, bundle)
            for bundle in config.oecd.icio.bundles
        ]

        # Detect whether these are SML-style bundles (per-year wide IO tables).
        year_members: dict[int, tuple[Path, str]] = {}
        for zip_path in bundle_zips:
            for year, member in _zip_members_by_year(zip_path).items():
                year_members.setdefault(year, (zip_path, member))

        target_years = list(range(config.years.base_start, config.years.end + 1))
        base_years = set(range(config.years.base_start, config.years.base_end + 1))

        countries_path = paths.resolve_project_path(config.countries.include_csv)
        dest_countries = set(pd.read_csv(countries_path)["iso3"].dropna().astype(str))
        dest_countries -= set(config.countries.exclude_iso3 or [])

        if year_members:
            logger.info("Building ICIO panels from SML wide tables (manual ZIPs)")
            output_rows: list[pd.DataFrame] = []
            hhi_rows: list[pd.DataFrame] = []

            base_sum: np.ndarray | None = None
            base_count = 0
            sector_list: list[str] | None = None
            dest_labels: list[str] | None = None

            for year in target_years:
                item = year_members.get(year)
                if item is None:
                    raise FileNotFoundError(
                        f"Missing ICIO SML CSV for year {year} in {raw_dir}. "
                        f"Found years: {sorted(year_members.keys())}"
                    )
                zip_path, member = item
                logger.info("Processing ICIO SML %s (%s)", year, zip_path.name)
                out_va, out_hhi, shares_sector, sectors, labels = _process_sml_year(
                    zip_path=zip_path,
                    member=member,
                    year=year,
                    dest_countries=dest_countries,
                )
                output_rows.append(out_va)
                hhi_rows.append(out_hhi)

                if year in base_years:
                    if base_sum is None:
                        base_sum = shares_sector.copy()
                        sector_list = sectors
                        dest_labels = labels
                    else:
                        base_sum += shares_sector
                    base_count += 1

            if base_sum is None or base_count == 0 or sector_list is None or dest_labels is None:
                raise RuntimeError(
                    "Base-year IO shares could not be computed (no base years processed). "
                    f"Expected base years {sorted(base_years)}."
                )

            base_avg = base_sum / float(base_count)

            # Build base IO shares long table.
            dest_country = [lab.split("|", 1)[0] for lab in dest_labels]
            dest_sector = [lab.split("|", 1)[1] for lab in dest_labels]
            n_dest = len(dest_labels)
            n_sectors = len(sector_list)

            shares = pd.DataFrame(
                {
                    "country_iso3": np.repeat(dest_country, n_sectors),
                    "downstream_icio50": np.repeat(dest_sector, n_sectors),
                    "upstream_icio50": np.tile(sector_list, n_dest),
                    "ioshare_base": base_avg.T.reshape(-1),
                }
            )

            output_va = pd.concat(output_rows, ignore_index=True)
            hhi = pd.concat(hhi_rows, ignore_index=True)
        else:
            # Long-format bundles: extract and concatenate.
            extract_dir = raw_dir / "extracted"
            csv_paths: list[Path] = []
            for zip_path in bundle_zips:
                csv_paths.append(_extract_first_csv(zip_path, extract_dir))

            years = range(config.years.base_start, config.years.end + 1)
            lazy_frames = [read_icio_long(path, years) for path in csv_paths]
            df = pl.concat(lazy_frames).collect()

            output_va = compute_output_va(df)
            shares = compute_input_shares_base(
                df, range(config.years.base_start, config.years.base_end + 1)
            )
            hhi = compute_import_hhi(df)

    output_va_df = output_va.copy()
    base_years = range(config.years.base_start, config.years.base_end + 1)
    base_va = output_va_df[output_va_df["year"].isin(base_years)]
    split_rows = []
    for nace_code, icio_list in {"24": ["C24A", "C24B"], "30": ["C301", "C302T309"]}.items():
        subset = base_va[base_va["icio50"].isin(icio_list)]
        total = subset["gross_output"].sum()
        for icio in icio_list:
            weight = 0.5 if total == 0 else subset.loc[subset["icio50"] == icio, "gross_output"].sum() / total
            split_rows.append({"nace_code": nace_code, "icio50": icio, "weight": weight})
    split_weights = pd.DataFrame(split_rows)

    write_parquet(output_va, output_path)
    write_parquet(shares, shares_path)
    write_parquet(hhi, hhi_path)
    write_parquet(split_weights, split_path)
    logger.info("Wrote ICIO panels to %s", paths.data_int)
    record_manifest(
        paths,
        config.model_dump(),
        "oecd_build_icio_panels",
        [paths.data_raw / "oecd_icio"],
        [output_path, shares_path, hhi_path, split_path],
    )
