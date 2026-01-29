"""Build a local DuckDB database from manually downloaded State Aid portal CSVs."""
from __future__ import annotations

import gzip
from pathlib import Path

import duckdb

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths


def _iter_csv_files(raw_dir: Path) -> list[Path]:
    paths: list[Path] = []
    candidates: list[Path] = []
    candidates.extend(raw_dir.rglob("*.csv"))
    candidates.extend(raw_dir.rglob("*.csv.gz"))
    for csv_path in candidates:
        try:
            rel = csv_path.relative_to(raw_dir)
        except ValueError:
            rel = csv_path
        if any(part.startswith(("_", ".")) for part in rel.parts):
            continue
        if csv_path.name.endswith(".csv.partial"):
            continue
        if csv_path.name.endswith(".csv.gz.partial"):
            continue
        paths.append(csv_path)
    return sorted(paths)


def _looks_like_portal_export(csv_path: Path) -> bool:
    try:
        if csv_path.suffix == ".gz" or csv_path.name.endswith(".csv.gz"):
            header = gzip.open(csv_path, "rt", encoding="utf-8", errors="replace").readline().lower()
        else:
            header = csv_path.open("r", encoding="utf-8", errors="replace").readline().lower()
    except OSError:
        return False
    return ("date of granting" in header) and ("name of the beneficiary" in header)


def _pick_column(available: set[str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in available:
            return cand
    return None


def _copy_query_to_csv(
    con: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
    *,
    overwrite: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and destination.exists():
        destination.unlink()
    dest_escaped = str(destination).replace("'", "''")
    con.execute(f"COPY ({query}) TO '{dest_escaped}' (HEADER, DELIMITER ',')")


def _sql_list(values: list[str]) -> str:
    escaped = [value.replace("'", "''") for value in values]
    return ", ".join(f"'{value}'" for value in escaped)


def build_stateaid_db(config: AppConfig, force: bool = False) -> None:
    """Ingest State Aid portal CSV exports into a DuckDB DB and print basic stats."""

    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    raw_dir = paths.data_raw / "stateaid"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory {raw_dir}.")

    all_csvs = _iter_csv_files(raw_dir)
    portal_csvs = [p for p in all_csvs if _looks_like_portal_export(p)]
    if not portal_csvs:
        raise FileNotFoundError(
            f"No portal-export CSVs found under {raw_dir}. "
            "Expected headers like 'Date of granting' and 'Name of the beneficiary'."
        )

    db_path = paths.data_int / "stateaid.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        countries_path = paths.resolve_project_path(config.countries.include_csv)
        con.execute(
            """
            CREATE OR REPLACE TABLE country_lookup AS
            SELECT
              iso3,
              name,
              lower(name) AS name_lc
            FROM read_csv_auto(?, all_varchar=true)
            """,
            [str(countries_path)],
        )

        file_list = [str(p) for p in portal_csvs]
        if force:
            con.execute("DROP TABLE IF EXISTS stateaid_portal_raw")
            con.execute("DROP TABLE IF EXISTS stateaid_portal_norm")

        con.execute(
            """
            CREATE OR REPLACE TABLE stateaid_portal_raw AS
            SELECT *
            FROM read_csv_auto(
              ?,
              all_varchar=true,
              union_by_name=true,
              normalize_names=true,
              filename=true,
              ignore_errors=true
            )
            """,
            [file_list],
        )

        cols = [row[1] for row in con.execute("PRAGMA table_info('stateaid_portal_raw')").fetchall()]
        available = set(cols)

        def need(name: str, candidates: list[str]) -> str:
            col = _pick_column(available, candidates)
            if not col:
                raise RuntimeError(
                    f"Unable to locate '{name}' column in portal exports. "
                    f"Available columns: {sorted(available)}"
                )
            return col

        col_country = need("country", ["country"])
        col_ref_no = need("ref_no", ["ref_no", "ref_no_", "refno"])
        col_sa_number = _pick_column(available, ["sa_number", "sa_number_", "sanumber"])
        col_beneficiary = need(
            "beneficiary_name",
            ["name_of_the_beneficiary", "beneficiary_name", "name_of_beneficiary"],
        )
        col_sector = need(
            "sector_label",
            ["sector_nace", "sector_nace_", "sector_nace__"],
        )
        col_date = need(
            "granting_date",
            ["date_of_granting", "date_granting", "date_granted", "granting_date"],
        )
        col_currency = _pick_column(available, ["currency"])
        col_aid_element = _pick_column(
            available,
            [
                "aid_element_expressed_as_full_amount",
                "aid_element_expressed_as_full_amount_",
                "aid_element_expressed_as_full_amount__",
            ],
        )
        col_nominal = _pick_column(
            available,
            [
                "nominal_amount_expressed_as_full_amount",
                "nominal_amount_expressed_as_full_amount_",
                "nominal_amount_expressed_as_full_amount__",
            ],
        )
        col_aid_instrument = _pick_column(available, ["aid_instrument"])
        col_objective = _pick_column(available, ["objectives_of_the_aid", "objective"])

        col_region = _pick_column(available, ["region"])
        col_aid_measure_title = _pick_column(available, ["aid_measure_title"])

        def sql_ident(name: str | None) -> str:
            if not name:
                return "NULL"
            return f'"{name}"'

        def sql_text(name: str | None) -> str:
            return f"NULLIF(TRIM({sql_ident(name)}), '')"

        try:
            con.execute("SELECT try_strptime('01/01/2020', '%d/%m/%Y')")
            date_expr = lambda col_sql: f"try_strptime({col_sql}, '%d/%m/%Y')::DATE"
        except Exception:
            date_expr = lambda col_sql: (
                f"CASE WHEN {col_sql} ~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}$' "
                f"THEN strptime({col_sql}, '%d/%m/%Y')::DATE ELSE NULL END"
            )

        def amount_expr(col_sql: str) -> str:
            # Strip thousands separators, currency codes, NBSP/BOM, etc.
            return (
                "try_cast("
                f"NULLIF(regexp_replace({col_sql}, '[^0-9\\\\.-]', '', 'g'), '') AS DOUBLE)"
            )

        granting_date_sql = date_expr(sql_text(col_date))
        published_date_col = _pick_column(available, ["published_date"])
        published_date_sql = date_expr(sql_text(published_date_col)) if published_date_col else "NULL"

        aid_element_sql = amount_expr(sql_text(col_aid_element)) if col_aid_element else "NULL"
        nominal_sql = amount_expr(sql_text(col_nominal)) if col_nominal else "NULL"

        con.execute(
            f"""
            CREATE OR REPLACE TABLE stateaid_portal_norm AS
            SELECT
              filename AS source_file,
              {sql_text(col_country)} AS country_name,
              cl.iso3 AS country_iso3,
              {sql_text(col_ref_no)} AS ref_no,
              {sql_text(col_aid_measure_title)} AS aid_measure_title,
              {sql_text(col_sa_number)} AS sa_number,
              {sql_text(col_beneficiary)} AS beneficiary_name,
              {sql_text(col_region)} AS region,
              {sql_text(col_sector)} AS sector_label,
              {sql_text(col_aid_instrument)} AS aid_instrument,
              {sql_text(col_objective)} AS aid_objective,
              {sql_text(col_currency)} AS currency,
              {granting_date_sql} AS granting_date,
              EXTRACT(year FROM {granting_date_sql})::INTEGER AS year,
              {aid_element_sql} AS aid_element_amount,
              {nominal_sql} AS nominal_amount,
              {published_date_sql} AS published_date
            FROM stateaid_portal_raw r
            LEFT JOIN country_lookup cl
              ON lower({sql_text(col_country)}) = cl.name_lc
            """,
        )

        total = con.execute("SELECT COUNT(*) FROM stateaid_portal_norm").fetchone()[0]
        n_files = con.execute("SELECT COUNT(DISTINCT source_file) FROM stateaid_portal_norm").fetchone()[
            0
        ]
        missing_iso3 = con.execute(
            "SELECT COUNT(*) FROM stateaid_portal_norm WHERE country_iso3 IS NULL"
        ).fetchone()[0]

        logger.info("State aid DB ready: %s", db_path)
        logger.info("Portal rows: %s (from %s CSV files)", f"{total:,}", n_files)
        if missing_iso3:
            logger.warning("Rows with unmapped country -> ISO3: %s", f"{missing_iso3:,}")

        by_year = con.execute(
            """
            SELECT year, COUNT(*) AS rows
            FROM stateaid_portal_norm
            WHERE year IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()
        logger.info("Rows by year:\n%s", by_year.to_string(index=False))

        by_country = con.execute(
            """
            SELECT country_iso3, COUNT(*) AS rows
            FROM stateaid_portal_norm
            GROUP BY 1
            ORDER BY rows DESC
            """
        ).fetchdf()
        logger.info("Rows by country (top 20):\n%s", by_country.head(20).to_string(index=False))

        by_country_year = con.execute(
            """
            SELECT country_iso3, year, COUNT(*) AS rows
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NOT NULL AND year IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
        logger.info("Rows by country-year:\n%s", by_country_year.to_string(index=False))

        if col_currency:
            currencies = con.execute(
                """
                SELECT currency, COUNT(*) AS rows
                FROM stateaid_portal_norm
                GROUP BY 1
                ORDER BY rows DESC
                """
            ).fetchdf()
            logger.info("Currencies in raw exports:\n%s", currencies.to_string(index=False))

        checks_dir = paths.data_int / "stateaid_checks"
        checks_dir.mkdir(parents=True, exist_ok=True)

        # Clear old generated reports so removed checks don't linger.
        for report_path in checks_dir.glob("*.csv"):
            report_path.unlink()

        excluded_iso3 = list(config.countries.exclude_iso3 or [])
        excluded_sql = _sql_list(excluded_iso3)
        excluded_filter = f"AND country_iso3 NOT IN ({excluded_sql})" if excluded_iso3 else ""
        excluded_country_lookup_filter = f"AND iso3 NOT IN ({excluded_sql})" if excluded_iso3 else ""

        year_start = int(config.years.start)
        year_end = int(config.years.end)

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW stateaid_expected_countries AS
            SELECT iso3 AS country_iso3
            FROM country_lookup
            WHERE iso3 IS NOT NULL
              {excluded_country_lookup_filter}
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW stateaid_expected_years AS
            SELECT range AS year
            FROM range({year_start}, {year_end + 1})
            """
        )

        # Coverage and QC checks for manual downloads.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_coverage_country_year AS
            SELECT
              country_iso3,
              year,
              COUNT(*) AS rows,
              MIN(granting_date) AS min_date,
              MAX(granting_date) AS max_date,
              COUNT(DISTINCT granting_date) AS n_distinct_dates,
              COUNT(DISTINCT EXTRACT(month FROM granting_date))::INTEGER AS n_months_present,
              COUNT(DISTINCT source_file) AS n_source_files
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NOT NULL
              AND granting_date IS NOT NULL
              AND year BETWEEN {year_start} AND {year_end}
              {excluded_filter}
            GROUP BY 1, 2
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_missing_country_years AS
            WITH expected AS (
              SELECT c.country_iso3, y.year
              FROM stateaid_expected_countries c
              CROSS JOIN stateaid_expected_years y
            ),
            actual AS (
              SELECT country_iso3, year, COUNT(*) AS rows
              FROM stateaid_portal_norm
              WHERE country_iso3 IS NOT NULL
                AND granting_date IS NOT NULL
                AND year BETWEEN {year_start} AND {year_end}
                {excluded_filter}
              GROUP BY 1, 2
            )
            SELECT e.country_iso3, e.year
            FROM expected e
            LEFT JOIN actual a USING(country_iso3, year)
            WHERE a.rows IS NULL
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_duplicate_ref_no AS
            SELECT
              country_iso3,
              ref_no,
              COUNT(*) AS rows,
              COUNT(DISTINCT source_file) AS n_source_files,
              string_agg(DISTINCT source_file, ' | ' ORDER BY source_file) AS source_files,
              MIN(granting_date) AS min_date,
              MAX(granting_date) AS max_date
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NOT NULL
              AND granting_date IS NOT NULL
              AND year BETWEEN {year_start} AND {year_end}
              AND ref_no IS NOT NULL
              AND TRIM(ref_no) <> ''
              {excluded_filter}
            GROUP BY 1, 2
            HAVING COUNT(*) > 1
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_file_segments_country_year AS
            SELECT
              country_iso3,
              year,
              source_file,
              COUNT(*) AS rows,
              MIN(granting_date) AS min_date,
              MAX(granting_date) AS max_date
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NOT NULL
              AND granting_date IS NOT NULL
              AND year BETWEEN {year_start} AND {year_end}
              {excluded_filter}
            GROUP BY 1, 2, 3
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_global_boundary_gaps AS
            WITH bounds AS (
              SELECT
                MIN(granting_date) AS min_date,
                MAX(granting_date) AS max_date
              FROM stateaid_portal_norm
              WHERE country_iso3 IS NOT NULL
                AND granting_date IS NOT NULL
                AND year BETWEEN {year_start} AND {year_end}
                {excluded_filter}
            ),
            expected AS (
              SELECT
                make_date({year_start}, 1, 1) AS expected_start,
                make_date({year_end}, 12, 31) AS expected_end
            )
            SELECT
              'start' AS gap_type,
              expected_start,
              expected_end,
              min_date AS observed_min_date,
              max_date AS observed_max_date,
              expected_start AS gap_start,
              CAST(min_date - INTERVAL '1 day' AS DATE) AS gap_end,
              datediff('day', expected_start, min_date) AS gap_days
            FROM bounds, expected
            WHERE min_date IS NOT NULL
              AND min_date > expected_start

            UNION ALL

            SELECT
              'end' AS gap_type,
              expected_start,
              expected_end,
              min_date AS observed_min_date,
              max_date AS observed_max_date,
              CAST(max_date + INTERVAL '1 day' AS DATE) AS gap_start,
              expected_end AS gap_end,
              datediff('day', max_date, expected_end) AS gap_days
            FROM bounds, expected
            WHERE max_date IS NOT NULL
              AND max_date < expected_end
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_global_date_gaps AS
            WITH dates AS (
              SELECT DISTINCT granting_date
              FROM stateaid_portal_norm
              WHERE country_iso3 IS NOT NULL
                AND granting_date IS NOT NULL
                AND year BETWEEN {year_start} AND {year_end}
                {excluded_filter}
            ),
            ordered AS (
              SELECT
                granting_date,
                LAG(granting_date) OVER (ORDER BY granting_date) AS prev_date
              FROM dates
            )
            SELECT
              prev_date,
              granting_date AS next_date,
              datediff('day', prev_date, granting_date) - 1 AS gap_days,
              CAST(prev_date + INTERVAL '1 day' AS DATE) AS gap_start,
              CAST(granting_date - INTERVAL '1 day' AS DATE) AS gap_end
            FROM ordered
            WHERE prev_date IS NOT NULL
              AND datediff('day', prev_date, granting_date) > 1
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE stateaid_check_global_date_gaps_ge7d AS
            SELECT *
            FROM stateaid_check_global_date_gaps
            WHERE gap_days >= 7
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE stateaid_check_global_file_segments AS
            SELECT
              source_file,
              COUNT(*) AS rows,
              COUNT(DISTINCT country_iso3) AS n_countries,
              MIN(granting_date) AS min_date,
              MAX(granting_date) AS max_date
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NOT NULL
              AND granting_date IS NOT NULL
              AND year BETWEEN {year_start} AND {year_end}
              {excluded_filter}
            GROUP BY 1
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE stateaid_check_global_file_segment_gaps AS
            WITH ordered AS (
              SELECT
                source_file,
                min_date,
                max_date,
                LAG(max_date) OVER (ORDER BY min_date, max_date, source_file) AS prev_max_date,
                LAG(source_file) OVER (ORDER BY min_date, max_date, source_file) AS prev_source_file
              FROM stateaid_check_global_file_segments
            )
            SELECT
              prev_source_file,
              prev_max_date,
              source_file AS next_source_file,
              min_date AS next_min_date,
              datediff('day', prev_max_date, min_date) - 1 AS gap_days,
              CAST(prev_max_date + INTERVAL '1 day' AS DATE) AS gap_start,
              CAST(min_date - INTERVAL '1 day' AS DATE) AS gap_end
            FROM ordered
            WHERE prev_max_date IS NOT NULL
              AND datediff('day', prev_max_date, min_date) > 1
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE stateaid_check_unmapped_country_names AS
            SELECT
              country_name,
              COUNT(*) AS rows
            FROM stateaid_portal_norm
            WHERE country_iso3 IS NULL
              AND country_name IS NOT NULL
              AND TRIM(country_name) <> ''
            GROUP BY 1
            ORDER BY rows DESC
            """
        )

        # Persist reports for manual inspection.
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_coverage_country_year ORDER BY country_iso3, year",
            checks_dir / "coverage_country_year.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_missing_country_years ORDER BY country_iso3, year",
            checks_dir / "missing_country_years.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_duplicate_ref_no "
            "ORDER BY n_source_files DESC, rows DESC, country_iso3, ref_no",
            checks_dir / "duplicate_ref_no.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_file_segments_country_year "
            "ORDER BY country_iso3, year, min_date, max_date, source_file",
            checks_dir / "file_segments_country_year.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_global_boundary_gaps ORDER BY gap_type",
            checks_dir / "global_boundary_gaps.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_global_date_gaps ORDER BY gap_days DESC, gap_start",
            checks_dir / "global_date_gaps.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_global_date_gaps_ge7d ORDER BY gap_days DESC, gap_start",
            checks_dir / "global_date_gaps_ge7d.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_global_file_segments ORDER BY min_date, max_date, source_file",
            checks_dir / "global_file_segments.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_global_file_segment_gaps ORDER BY gap_days DESC, gap_start",
            checks_dir / "global_file_segment_gaps.csv",
        )
        _copy_query_to_csv(
            con,
            "SELECT * FROM stateaid_check_unmapped_country_names",
            checks_dir / "unmapped_country_names.csv",
        )

        missing_country_years = con.execute(
            "SELECT COUNT(*) FROM stateaid_check_missing_country_years"
        ).fetchone()[0]
        dup_ref_no = con.execute("SELECT COUNT(*) FROM stateaid_check_duplicate_ref_no").fetchone()[0]
        boundary_gaps = con.execute("SELECT COUNT(*) FROM stateaid_check_global_boundary_gaps").fetchone()[
            0
        ]
        date_gaps_ge7d = con.execute(
            "SELECT COUNT(*) FROM stateaid_check_global_date_gaps_ge7d"
        ).fetchone()[0]

        logger.info("Wrote state aid coverage checks to %s", checks_dir)
        if missing_country_years:
            logger.warning(
                "Missing country-years (no rows): %s (see %s)",
                f"{missing_country_years:,}",
                checks_dir / "missing_country_years.csv",
            )
        if dup_ref_no:
            logger.warning(
                "Duplicate Ref-no. detected (overlaps/duplicates): %s (see %s). "
                "Downstream `rp stateaid clean` deduplicates by Ref-no./award id.",
                f"{dup_ref_no:,}",
                checks_dir / "duplicate_ref_no.csv",
            )
        if boundary_gaps:
            logger.warning(
                "Boundary gaps in overall time horizon: %s (see %s)",
                f"{boundary_gaps:,}",
                checks_dir / "global_boundary_gaps.csv",
            )
        if date_gaps_ge7d:
            logger.warning(
                "Global date gaps >= 7 days (possible missing downloads): %s (see %s)",
                f"{date_gaps_ge7d:,}",
                checks_dir / "global_date_gaps_ge7d.csv",
            )
    finally:
        con.close()
