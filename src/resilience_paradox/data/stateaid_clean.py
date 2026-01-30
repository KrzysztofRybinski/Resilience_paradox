"""Clean State Aid awards into standardized parquet."""
from __future__ import annotations

import hashlib
import re

import duckdb
import numpy as np
import pandas as pd
import requests

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.io import write_parquet

EUSA_COLUMN_MAP = {
    "Member state": "country_iso3",
    "Granting date": "granting_date",
    "Beneficiary": "beneficiary_name",
    "Aid element (EUR)": "aid_amount_eur",
    "Aid instrument": "aid_instrument",
    "Aid objective": "aid_objective",
    "NACE code": "nace_code",
    "NUTS2": "nuts2",
    "Measure ID": "measure_id",
    "Case ID": "case_id",
}

PORTAL_COLUMN_MAP = {
    "Country": "country_name",
    "Ref-no.": "ref_no",
    "SA.Number": "case_id",
    "Name of the beneficiary": "beneficiary_name",
    "Aid Instrument": "aid_instrument",
    "Objectives of the Aid": "aid_objective",
    "Sector (NACE)": "sector_label",
    "Region": "nuts2",
    "Date of granting": "granting_date",
    "Aid element, expressed as full amount": "aid_amount_raw",
    "Nominal Amount, expressed as full amount": "nominal_amount_raw",
    "Currency": "currency",
}

NACE_VOCAB_URL = "https://dd.eionet.europa.eu/vocabulary/eurostat/nace_r2/csv"


def _hash_row(row: pd.Series) -> str:
    payload = "|".join(str(value) for value in row.values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_amount(value: str | float | int) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(" ", "").replace(",", "")
    if "-" in cleaned or "–" in cleaned:
        parts = re.split(r"[-–]", cleaned)
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _normalize_text(value: str) -> str:
    cleaned = str(value or "").replace("\ufeff", "").replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def _notation_to_nace_code(notation: str) -> str | None:
    digits = "".join(ch for ch in str(notation) if ch.isdigit())
    if len(digits) < 2:
        return None
    if len(digits) == 2:
        return digits
    if len(digits) == 3:
        return f"{digits[:2]}.{digits[2:]}"
    return f"{digits[:2]}.{digits[2:4]}"


def _load_nace_label_mapping(paths: Paths) -> dict[str, str]:
    cache_path = paths.data_int / "nace_r2_vocabulary.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        resp = requests.get(NACE_VOCAB_URL, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    def read_vocab(sep: str | None) -> pd.DataFrame:
        kwargs: dict[str, object] = {"dtype": str, "keep_default_na": False}
        if sep is not None:
            kwargs["sep"] = sep
        return pd.read_csv(cache_path, **kwargs)  # type: ignore[arg-type]

    df = read_vocab(sep=None)

    def find_cols(frame: pd.DataFrame) -> tuple[str | None, str | None]:
        cols = {c.lower(): c for c in frame.columns}
        label_col = cols.get("label") or cols.get("preflabel")
        notation_col = cols.get("notation")
        if not label_col or not notation_col:
            label_col = next((c for c in frame.columns if "label" in c.lower()), None)
            notation_col = next((c for c in frame.columns if "notation" in c.lower()), None)
        return label_col, notation_col

    label_col, notation_col = find_cols(df)
    if not label_col or not notation_col:
        df = read_vocab(sep=";")
        label_col, notation_col = find_cols(df)
    if not label_col or not notation_col:
        raise RuntimeError(
            f"Unable to parse NACE vocabulary CSV at {cache_path}; expected Label/Notation columns."
        )

    mapping: dict[str, str] = {}
    for _, row in df[[label_col, notation_col]].iterrows():  # type: ignore[index]
        label = _normalize_text(row[label_col])
        code = _notation_to_nace_code(row[notation_col])
        if label and code and label not in mapping:
            mapping[label] = code
    return mapping


def _iter_stateaid_csvs(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for csv_path in raw_dir.rglob("*.csv"):
        rel = csv_path.relative_to(raw_dir)
        if any(part.startswith(("_", ".")) for part in rel.parts):
            continue
        if csv_path.name.endswith(".csv.partial"):
            continue
        files.append(csv_path)
    return sorted(files)


def _has_portal_db(paths: Paths) -> bool:
    db_path = paths.data_int / "stateaid.duckdb"
    if not db_path.exists():
        return False
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            return (
                con.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_name = 'stateaid_portal_norm'
                    """
                ).fetchone()[0]
                > 0
            )
        finally:
            con.close()
    except Exception:
        return False


def _warn_if_manual_export_checks_fail(paths: Paths, logger) -> None:
    checks_dir = paths.data_int / "stateaid_checks"
    if not checks_dir.exists():
        logger.warning(
            "No state aid coverage checks found at %s. Run `rp stateaid build-db` to generate "
            "missing/overlap reports.",
            checks_dir,
        )
        return

    checks: list[tuple[str, str]] = [
        ("missing_country_years.csv", "Missing country-years (no rows)"),
        ("duplicate_ref_no.csv", "Duplicate Ref-no. (overlaps/duplicates)"),
        ("global_boundary_gaps.csv", "Boundary gaps in overall time horizon"),
        ("global_date_gaps_ge7d.csv", "Global date gaps >= 7 days"),
        ("global_file_segment_gaps.csv", "Gaps between file segments (global, heuristic)"),
        ("unmapped_country_names.csv", "Unmapped country names"),
    ]

    for filename, label in checks:
        report_path = checks_dir / filename
        if not report_path.exists():
            continue
        try:
            sample = pd.read_csv(report_path, nrows=20)
        except Exception as exc:
            logger.warning("Could not read %s: %s", report_path, exc)
            continue
        if sample.empty:
            continue
        logger.warning("%s detected; see %s (showing up to 20 rows)", label, report_path)
        logger.warning("%s", sample.to_string(index=False))


def clean_stateaid(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    output_path = paths.data_int / "stateaid_awards.parquet"
    if output_path.exists() and not force:
        logger.info("State aid awards already cleaned; skipping.")
        return

    raw_dir = paths.data_raw / "stateaid"

    if not sample and _has_portal_db(paths):
        db_path = paths.data_int / "stateaid.duckdb"
        logger.info("Cleaning from DuckDB portal table: %s", db_path)
        _warn_if_manual_export_checks_fail(paths, logger)

        nace_map = _load_nace_label_mapping(paths)

        con = duckdb.connect(str(db_path))
        try:
            from resilience_paradox.data.prices import load_fx_annual, load_hicp_deflator

            sector_labels = con.execute(
                "SELECT DISTINCT sector_label FROM stateaid_portal_norm WHERE sector_label IS NOT NULL"
            ).fetchdf()
            sector_labels["sector_label_norm"] = sector_labels["sector_label"].map(_normalize_text)
            sector_labels["nace_code"] = sector_labels["sector_label_norm"].map(nace_map).fillna("")
            mapping_df = sector_labels[["sector_label_norm", "nace_code"]].drop_duplicates()
            con.register("sector_to_nace", mapping_df)

            fx_annual = load_fx_annual(paths, config)
            hicp_deflator = load_hicp_deflator(paths, config)
            con.register("fx_annual", fx_annual)
            con.register("hicp_deflator", hicp_deflator)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                output_path.unlink()

            excluded = [iso3 for iso3 in config.countries.exclude_iso3]
            excluded_sql = ", ".join(f"'{iso3}'" for iso3 in excluded)
            year_start = int(config.years.start)
            year_end = int(config.years.end)

            checks_dir = paths.data_int / "stateaid_checks"
            checks_dir.mkdir(parents=True, exist_ok=True)

            award_id_expr = (
                "COALESCE("
                "NULLIF(ref_no, ''), "
                "md5(coalesce(country_iso3, '') || '|' || coalesce(beneficiary_name, '') || '|' "
                "|| coalesce(cast(granting_date as varchar), '') || '|' "
                "|| coalesce(cast(coalesce(aid_element_amount, nominal_amount) as varchar), ''))"
                ")"
            )

            # Report missing conversion factors (do not drop rows).
            missing_fx_path = checks_dir / "missing_fx_rates.csv"
            missing_deflator_path = checks_dir / "missing_hicp_deflator_years.csv"
            missing_fx_escaped = str(missing_fx_path).replace("'", "''")
            missing_deflator_escaped = str(missing_deflator_path).replace("'", "''")
            currency_rows_path = checks_dir / "currency_rows_pre_conversion.csv"
            currency_rows_escaped = str(currency_rows_path).replace("'", "''")
            conversion_summary_path = checks_dir / "currency_conversion_summary.csv"
            conversion_summary_escaped = str(conversion_summary_path).replace("'", "''")

            diagnostic_base = f"""
            WITH base AS (
              SELECT
                r.country_iso3,
                r.year,
                COALESCE(NULLIF(upper(trim(r.currency)), ''), 'EUR') AS aid_currency,
                coalesce(r.aid_element_amount, r.nominal_amount) AS amount
              FROM stateaid_portal_norm r
              WHERE r.country_iso3 IS NOT NULL
                AND r.country_iso3 NOT IN ({excluded_sql})
                AND r.year BETWEEN {year_start} AND {year_end}
                AND r.granting_date IS NOT NULL
                AND coalesce(r.aid_element_amount, r.nominal_amount) IS NOT NULL
            )
            """

            currency_totals = con.execute(
                f"""
                {diagnostic_base}
                SELECT
                  aid_currency AS currency,
                  COUNT(*) AS rows
                FROM base
                GROUP BY 1
                ORDER BY rows DESC
                """
            ).fetchdf()
            if not currency_totals.empty:
                logger.info(
                    "State aid rows by original currency (pre-conversion, non-null amounts):\n%s",
                    currency_totals.to_string(index=False),
                )

            ltl_rows = con.execute(
                f"""
                {diagnostic_base}
                SELECT COUNT(*) FROM base WHERE aid_currency = 'LTL'
                """
            ).fetchone()[0]
            if ltl_rows:
                logger.info(
                    "Detected %s State Aid rows with currency LTL. Conversion uses the fixed irrevocable rate 1 EUR = 3.45280 LTL.",
                    ltl_rows,
                )

            con.execute(
                f"""
                COPY (
                  {diagnostic_base}
                  SELECT
                    year,
                    aid_currency AS currency,
                    COUNT(*) AS rows
                  FROM base
                  GROUP BY 1,2
                  ORDER BY 1,2
                ) TO '{currency_rows_escaped}' (HEADER, DELIMITER ',')
                """
            )

            total_rows, eur_rows, local_rows, converted_local_rows, missing_fx_preview = con.execute(
                f"""
                {diagnostic_base}
                SELECT
                  COUNT(*) AS rows_total,
                  SUM(CASE WHEN aid_currency = 'EUR' THEN 1 ELSE 0 END) AS rows_eur_currency,
                  SUM(CASE WHEN aid_currency <> 'EUR' THEN 1 ELSE 0 END) AS rows_local_currency,
                  SUM(CASE WHEN aid_currency <> 'EUR' AND fx.rate_per_eur IS NOT NULL THEN 1 ELSE 0 END) AS rows_converted_from_local_currency,
                  SUM(CASE WHEN fx.rate_per_eur IS NULL THEN 1 ELSE 0 END) AS rows_missing_fx
                FROM base
                LEFT JOIN fx_annual fx
                  ON fx.year = base.year
                 AND fx.currency = base.aid_currency
                """
            ).fetchone()
            rows_with_eur_amount = int(total_rows) - int(missing_fx_preview)

            con.execute(
                f"""
                COPY (
                  {diagnostic_base}
                  SELECT
                    COUNT(*) AS rows_total,
                    SUM(CASE WHEN aid_currency = 'EUR' THEN 1 ELSE 0 END) AS rows_eur_currency,
                    SUM(CASE WHEN aid_currency <> 'EUR' THEN 1 ELSE 0 END) AS rows_local_currency,
                    SUM(CASE WHEN aid_currency <> 'EUR' AND fx.rate_per_eur IS NOT NULL THEN 1 ELSE 0 END) AS rows_converted_from_local_currency,
                    SUM(CASE WHEN fx.rate_per_eur IS NULL THEN 1 ELSE 0 END) AS rows_missing_fx,
                    SUM(CASE WHEN h.deflator_to_base IS NULL THEN 1 ELSE 0 END) AS rows_missing_deflator
                  FROM base
                  LEFT JOIN fx_annual fx
                    ON fx.year = base.year
                   AND fx.currency = base.aid_currency
                  LEFT JOIN hicp_deflator h
                    ON h.year = base.year
                ) TO '{conversion_summary_escaped}' (HEADER, DELIMITER ',')
                """
            )

            logger.info(
                "State aid conversion coverage: local-currency rows=%s; converted from local=%s; total rows with EUR amount=%s/%s. Reports: %s, %s",
                int(local_rows),
                int(converted_local_rows),
                rows_with_eur_amount,
                int(total_rows),
                currency_rows_path,
                conversion_summary_path,
            )

            missing_fx_count = con.execute(
                f"""
                {diagnostic_base}
                SELECT COUNT(*) FROM base
                LEFT JOIN fx_annual fx
                  ON fx.year = base.year
                 AND fx.currency = base.aid_currency
                WHERE fx.rate_per_eur IS NULL
                """
            ).fetchone()[0]

            missing_deflator_count = con.execute(
                f"""
                {diagnostic_base}
                SELECT COUNT(*) FROM base
                LEFT JOIN hicp_deflator h
                  ON h.year = base.year
                WHERE h.deflator_to_base IS NULL
                """
            ).fetchone()[0]

            con.execute(
                f"""
                COPY (
                  {diagnostic_base}
                  SELECT
                    base.year,
                    aid_currency AS currency,
                    COUNT(*) AS rows
                  FROM base
                  LEFT JOIN fx_annual fx
                    ON fx.year = base.year
                   AND fx.currency = base.aid_currency
                  WHERE fx.rate_per_eur IS NULL
                  GROUP BY 1,2
                  ORDER BY 1,2
                ) TO '{missing_fx_escaped}' (HEADER, DELIMITER ',')
                """
            )

            con.execute(
                f"""
                COPY (
                  {diagnostic_base}
                  SELECT
                    base.year,
                    COUNT(*) AS rows
                  FROM base
                  LEFT JOIN hicp_deflator h
                    ON h.year = base.year
                  WHERE h.deflator_to_base IS NULL
                  GROUP BY 1
                  ORDER BY 1
                ) TO '{missing_deflator_escaped}' (HEADER, DELIMITER ',')
                """
            )

            if missing_fx_count or missing_deflator_count:
                raise RuntimeError(
                    "Missing FX rates and/or HICP deflators for some State Aid rows. "
                    "I will not silently drop or zero-out these awards. "
                    "See:\n"
                    f"  - {missing_fx_path}\n"
                    f"  - {missing_deflator_path}\n"
                    "Fix by running `rp prices download --force` (ECB FX + Eurostat HICP) "
                    "or manually providing the missing series, then rerun `rp stateaid clean --force`."
                )

            query = f"""
            SELECT
              award_id,
              country_iso3,
              granting_date,
              year,
              beneficiary_name,
              aid_amount_original,
              aid_amount_source,
              fx_rate_per_eur,
              deflator_to_base,
              missing_fx,
              missing_deflator,
              aid_amount_eur,
              aid_amount_real_eur,
              aid_amount_eur_million,
              aid_amount_real_eur_million,
              aid_currency,
              aid_instrument,
              aid_objective,
              nace_code,
              nuts2,
              measure_id,
              case_id
            FROM (
              SELECT
                {award_id_expr} AS award_id,
                country_iso3,
                granting_date,
                r.year AS year,
                beneficiary_name,
                COALESCE(NULLIF(upper(trim(r.currency)), ''), 'EUR') AS aid_currency,
                coalesce(r.aid_element_amount, r.nominal_amount) AS aid_amount_original,
                CASE
                  WHEN r.aid_element_amount IS NOT NULL THEN 'aid_element'
                  ELSE 'nominal'
                END AS aid_amount_source,
                fx.rate_per_eur AS fx_rate_per_eur,
                h.deflator_to_base AS deflator_to_base,
                fx.rate_per_eur IS NULL AS missing_fx,
                h.deflator_to_base IS NULL AS missing_deflator,
                CASE
                  WHEN fx.rate_per_eur IS NULL THEN NULL
                  ELSE (coalesce(r.aid_element_amount, r.nominal_amount) / fx.rate_per_eur)
                END AS aid_amount_eur,
                CASE
                  WHEN fx.rate_per_eur IS NULL OR h.deflator_to_base IS NULL THEN NULL
                  ELSE ((coalesce(r.aid_element_amount, r.nominal_amount) / fx.rate_per_eur) * h.deflator_to_base)
                END AS aid_amount_real_eur,
                CASE
                  WHEN fx.rate_per_eur IS NULL THEN NULL
                  ELSE ((coalesce(r.aid_element_amount, r.nominal_amount) / fx.rate_per_eur) / 1e6)
                END AS aid_amount_eur_million,
                CASE
                  WHEN fx.rate_per_eur IS NULL OR h.deflator_to_base IS NULL THEN NULL
                  ELSE (((coalesce(r.aid_element_amount, r.nominal_amount) / fx.rate_per_eur) * h.deflator_to_base) / 1e6)
                END AS aid_amount_real_eur_million,
                aid_instrument,
                aid_objective,
                COALESCE(sector_to_nace.nace_code, '') AS nace_code,
                region AS nuts2,
                aid_measure_title AS measure_id,
                sa_number AS case_id,
                row_number() OVER (
                  PARTITION BY {award_id_expr}
                  ORDER BY source_file
                ) AS rn
              FROM stateaid_portal_norm r
              LEFT JOIN sector_to_nace
                ON lower(regexp_replace(trim(r.sector_label), '\\s+', ' ', 'g')) = sector_to_nace.sector_label_norm
              LEFT JOIN fx_annual fx
                ON fx.year = r.year
               AND fx.currency = COALESCE(NULLIF(upper(trim(r.currency)), ''), 'EUR')
              LEFT JOIN hicp_deflator h
                ON h.year = r.year
              WHERE r.country_iso3 IS NOT NULL
                AND r.country_iso3 NOT IN ({excluded_sql})
                AND r.year BETWEEN {year_start} AND {year_end}
                AND r.granting_date IS NOT NULL
                AND coalesce(r.aid_element_amount, r.nominal_amount) IS NOT NULL
            ) t
            WHERE rn = 1
            """

            output_escaped = str(output_path).replace("'", "''")
            con.execute(f"COPY ({query}) TO '{output_escaped}' (FORMAT PARQUET)")
        finally:
            con.close()

        logger.info("Wrote cleaned state aid awards to %s", output_path)
        logger.info("Wrote FX/HICP missingness reports to %s", paths.data_int / "stateaid_checks")
        record_manifest(
            paths,
            config.model_dump(),
            "stateaid_clean",
            [paths.data_raw / "stateaid", paths.data_int / "stateaid.duckdb"],
            [output_path],
        )
        return

    csv_files = _iter_stateaid_csvs(raw_dir)
    if not csv_files:
        raise FileNotFoundError(
            "No state aid CSV files found. Place portal exports under data/raw/stateaid "
            "(or run `rp stateaid build-db` first)."
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    if "Member state" in raw.columns:
        df = raw.rename(columns=EUSA_COLUMN_MAP)
        for col in EUSA_COLUMN_MAP.values():
            if col not in df.columns:
                df[col] = np.nan
        df["granting_date"] = pd.to_datetime(df["granting_date"], errors="coerce")
        df["year"] = df["granting_date"].dt.year
        df["aid_amount_eur"] = df["aid_amount_eur"].apply(_parse_amount)
        df = df.dropna(subset=["country_iso3", "aid_amount_eur"])
        df["award_id"] = df.apply(_hash_row, axis=1)
    elif "Date of granting" in raw.columns and "Name of the beneficiary" in raw.columns:
        df = raw.rename(columns=PORTAL_COLUMN_MAP)
        for col in PORTAL_COLUMN_MAP.values():
            if col not in df.columns:
                df[col] = np.nan
        countries_path = paths.resolve_project_path(config.countries.include_csv)
        country_lookup = pd.read_csv(countries_path, dtype=str).dropna()
        name_to_iso3 = {
            _normalize_text(row["name"]): row["iso3"] for _, row in country_lookup.iterrows()
        }
        nace_map = _load_nace_label_mapping(paths)
        df["nace_code"] = df["sector_label"].map(lambda x: nace_map.get(_normalize_text(x), ""))
        df["granting_date"] = pd.to_datetime(df["granting_date"], errors="coerce", dayfirst=True)
        df["year"] = df["granting_date"].dt.year
        df["aid_amount_eur"] = df["aid_amount_raw"].apply(_parse_amount)
        df.loc[df["aid_amount_eur"].isna(), "aid_amount_eur"] = df["nominal_amount_raw"].apply(
            _parse_amount
        )
        df["country_iso3"] = df["country_name"].map(lambda x: name_to_iso3.get(_normalize_text(x), np.nan))
        df["award_id"] = df["ref_no"].where(
            df["ref_no"].astype(str).str.strip().ne(""), df.apply(_hash_row, axis=1)
        )
    else:
        raise RuntimeError(
            "Unrecognized state aid CSV schema. Expected either EUSA columns "
            "('Member state', 'Granting date', ...) or portal export columns "
            "('Date of granting', 'Name of the beneficiary', ...)."
        )

    df = df[df["year"].between(config.years.start, config.years.end)]

    keep_cols = [
        "award_id",
        "country_iso3",
        "granting_date",
        "year",
        "beneficiary_name",
        "aid_amount_eur",
        "aid_instrument",
        "aid_objective",
        "nace_code",
        "nuts2",
        "measure_id",
        "case_id",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep_cols]
    df = df[~df["country_iso3"].isin(config.countries.exclude_iso3)]
    df = df.drop_duplicates(subset=["award_id"])

    write_parquet(df, output_path)
    logger.info("Wrote cleaned state aid awards to %s", output_path)
    record_manifest(
        paths,
        config.model_dump(),
        "stateaid_clean",
        [paths.data_raw / "stateaid"],
        [output_path],
    )
