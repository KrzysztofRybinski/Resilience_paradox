"""Download State Aid transparency data."""
from __future__ import annotations

import calendar
import csv
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest

TAM_URL = "https://webgate.ec.europa.eu/competition/transparency/public?lang=en"


class ExportRequiresPersonalData(RuntimeError):
    """Portal requires email-based export (no direct CSV download available)."""


class TransientPortalError(RuntimeError):
    """Portal returned a transient error page (e.g., internal server error)."""


class EmailExportNotAvailable(RuntimeError):
    """Email-based export controls are not available for this query/range."""


def _load_countries(path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def _download_eusa(destination: Path) -> None:
    api_url = "https://api.github.com/repos/jfjelstul/eusa/contents/data"
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    entries = response.json()
    target = next(
        (item for item in entries if "state_aid" in item["name"] and item["name"].endswith(".csv")),
        None,
    )
    if not target:
        raise RuntimeError("Unable to locate state aid CSV in jfjelstul/eusa repo")
    csv_url = target["download_url"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(csv_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def _download_playwright(
    country_iso3: str,
    country_name: str,
    year: int,
    destination: Path,
    headless: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless)
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise RuntimeError(
                    "Playwright Chromium executable is missing. Run `uv run playwright install chromium` "
                    "(or `uv run playwright install`) and re-run the pipeline."
                ) from exc
            raise
        context = browser.new_context()
        page = context.new_page()
        debug_dir = destination.parent / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        parts_dir = destination.parent / "_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        wrote_any = False

        def reset_session() -> None:
            nonlocal context, page
            try:
                context.close()
            except Exception:
                pass
            context = browser.new_context()
            page = context.new_page()

        def write_debug(prefix_override: str | None = None) -> None:
            prefix = prefix_override or f"{country_iso3}_{year}"
            try:
                page.screenshot(path=str(debug_dir / f"{prefix}.png"), full_page=True)
            except Exception:
                pass
            try:
                (debug_dir / f"{prefix}.html").write_text(page.content(), encoding="utf-8")
            except Exception:
                pass

        def dump_debug(reason: str) -> None:
            write_debug()
            raise RuntimeError(
                f"Playwright could not locate expected controls on the State Aid portal ({reason}). "
                f"Debug artifacts saved to {debug_dir}."
            )

        def click_if_visible(locator) -> bool:
            try:
                if locator.count() and locator.first.is_visible():
                    locator.first.click()
                    return True
            except Exception:
                return False
            return False

        def accept_cookies_if_present() -> None:
            for loc in [
                page.get_by_role("button", name=re.compile(r"accept all", re.I)),
                page.get_by_role("button", name=re.compile(r"accept", re.I)),
                page.get_by_role("button", name=re.compile(r"agree", re.I)),
                page.get_by_role("button", name=re.compile(r"ok", re.I)),
            ]:
                if click_if_visible(loc):
                    page.wait_for_timeout(500)
                    return

        def is_internal_error_page() -> bool:
            try:
                title = page.title()
                if title and re.search(r"internal application error", title, re.I):
                    return True
            except Exception:
                pass
            try:
                errors = page.locator("div.errors")
                if errors.count():
                    text = errors.first.inner_text()
                    if re.search(r"internal server error", text, re.I):
                        return True
            except Exception:
                pass
            return False

        def raise_if_internal_error(context: str) -> None:
            if not is_internal_error_page():
                return
            write_debug()
            raise TransientPortalError(context)

        def set_granting_country() -> None:
            # Current (v2.x) portal landing page uses country checkboxes with values like "CountryAUT".
            home_checkbox = page.locator(
                f"form#publicSearchForm input[type='checkbox'][name='countries'][value='Country{country_iso3}']"
            )
            if home_checkbox.count():
                try:
                    home_checkbox.first.check()
                    return
                except Exception:
                    pass

            # Older portal versions used a classic <select>.
            for sel in [
                "select[name='grantingCountry']",
                "select[name*='grantingCountry' i]",
                "select[id*='grantingCountry' i]",
            ]:
                loc = page.locator(sel)
                if not loc.count():
                    continue
                for value in [country_iso3, country_name]:
                    try:
                        if value:
                            loc.first.select_option(value)
                            return
                    except Exception:
                        continue

            # Newer portal versions may use inputs/checkboxes.
            for value in [country_name, country_iso3]:
                if not value:
                    continue
                checkbox = page.get_by_role(
                    "checkbox", name=re.compile(rf"^{re.escape(value)}$", re.I)
                )
                try:
                    if checkbox.count():
                        checkbox.first.check()
                        return
                except Exception:
                    pass

            dump_debug("granting country selector missing")

        def fill_date_range_for(start: date, end: date) -> None:
            start_str = start.strftime("%d/%m/%Y")
            end_str = end.strftime("%d/%m/%Y")

            def fill_first(selectors: list[str], value: str) -> bool:
                for sel in selectors:
                    loc = page.locator(sel)
                    try:
                        if loc.count():
                            loc.first.fill(value)
                            return True
                    except Exception:
                        continue
                return False

            ok_start = fill_first(
                [
                    "input[name='grantingDateFrom']",
                    "input[id='grantingDateFrom']",
                    "input[name*='grantingDateFrom' i]",
                    "input[id*='grantingDateFrom' i]",
                    "input[name='dateGrantedFrom']",
                    "input[id='dateGrantedFrom']",
                    "input[name*='dateGrantedFrom' i]",
                    "input[id*='dateGrantedFrom' i]",
                ],
                start_str,
            )
            ok_end = fill_first(
                [
                    "input[name='grantingDateTo']",
                    "input[id='grantingDateTo']",
                    "input[name*='grantingDateTo' i]",
                    "input[id*='grantingDateTo' i]",
                    "input[name='dateGrantedTo']",
                    "input[id='dateGrantedTo']",
                    "input[name*='dateGrantedTo' i]",
                    "input[id*='dateGrantedTo' i]",
                ],
                end_str,
            )

            if not (ok_start and ok_end):
                # Last resort: label-based lookup (language-dependent).
                try:
                    page.get_by_label(re.compile(r"granting date.*from", re.I)).fill(start_str)
                    page.get_by_label(re.compile(r"granting date.*to", re.I)).fill(end_str)
                    return
                except Exception:
                    dump_debug("date range inputs missing")

        def has_date_inputs() -> bool:
            return (
                page.locator(
                    "input[name='grantingDateFrom'], input[id='grantingDateFrom'], "
                    "input[name*='grantingDateFrom' i], input[id*='grantingDateFrom' i], "
                    "input[name='dateGrantedFrom'], input[id='dateGrantedFrom'], "
                    "input[name*='dateGrantedFrom' i], input[id*='dateGrantedFrom' i]"
                ).count()
                > 0
            )

        def close_modal_if_present() -> None:
            # Personal data authorization dialog (shown when export must be emailed).
            try:
                dialog = page.locator(".ui-dialog[aria-describedby='dialog-confirmation']")
                if dialog.count() and dialog.first.is_visible():
                    cancel = dialog.locator("button:has-text('CANCEL')")
                    if cancel.count() and cancel.first.is_visible():
                        cancel.first.click()
                        page.wait_for_timeout(250)
                        return
                    close_btn = dialog.locator("button.ui-dialog-titlebar-close")
                    if close_btn.count() and close_btn.first.is_visible():
                        close_btn.first.click()
                        page.wait_for_timeout(250)
                        return
            except Exception:
                pass

            # Download dialog from the direct-export flow.
            try:
                dialog = page.locator(".ui-dialog[aria-describedby='download-dialog']")
                if dialog.count() and dialog.first.is_visible():
                    close_btn = dialog.locator("button.ui-dialog-titlebar-close")
                    if close_btn.count() and close_btn.first.is_visible():
                        close_btn.first.click()
                        page.wait_for_timeout(250)
                        return
            except Exception:
                pass

        def go_back_to_search_form() -> None:
            if has_date_inputs():
                return
            raise_if_internal_error("portal internal server error while navigating to search form")
            close_modal_if_present()
            for loc in [
                page.locator("a.button-back"),
                page.get_by_role("link", name=re.compile(r"^back$", re.I)),
                page.locator("a.button-home"),
                page.get_by_role("link", name=re.compile(r"^home$", re.I)),
            ]:
                try:
                    if not loc.count():
                        continue
                    try:
                        with page.expect_navigation(timeout=60000):
                            loc.first.click()
                    except PlaywrightTimeoutError:
                        pass
                    page.wait_for_timeout(750)
                    if has_date_inputs():
                        return
                except Exception:
                    continue
            # Last resort: hard reset to the start.
            page.goto(TAM_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            raise_if_internal_error("portal internal server error while loading landing page")
            accept_cookies_if_present()
            set_granting_country()
            if not has_date_inputs():
                click_search(wait_for_navigation=True)
            if not has_date_inputs():
                dump_debug("unable to return to search form")

        def click_search(*, wait_for_navigation: bool = False) -> None:
            for loc in [
                page.locator("button#searchButton"),
                page.get_by_role("button", name=re.compile(r"^search$", re.I)),
                page.get_by_role("button", name=re.compile(r"search", re.I)),
            ]:
                try:
                    if loc.count():
                        if wait_for_navigation:
                            try:
                                with page.expect_navigation(timeout=60000):
                                    loc.first.click()
                            except PlaywrightTimeoutError:
                                # Some portal flows update results via XHR instead of full navigation.
                                pass
                        else:
                            loc.first.click()
                        page.wait_for_timeout(1500)
                        raise_if_internal_error("portal internal server error after submitting search")
                        return
                except Exception:
                    continue
            dump_debug("search button missing")

        def export_csv(output_path: Path) -> bool:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raise_if_internal_error("portal internal server error on results page")

            try:
                if page.locator("#resultsTable").count() and page.locator("#resultsTable tbody tr").count() == 0:
                    return False
            except Exception:
                pass

            # When results are very large, the portal switches to an email-based export flow
            # that requires providing personal data. We avoid that and instead split date ranges.
            if (
                page.locator("button#newUserCriteriaToExportCsv").count()
                and not page.locator("a.exportCSVLink[href*='format=CSV' i]").count()
                and not page.locator("button#exportCsvButton").count()
            ):
                raise ExportRequiresPersonalData

            export_trigger = None
            for loc in [
                page.locator("button#exportCsvButton"),
                page.locator("a.exportCSVLink[href*='format=CSV' i]"),
                page.get_by_role("link", name=re.compile(r"export\s+csv", re.I)),
                page.get_by_role("button", name=re.compile(r"export\s+csv", re.I)),
                page.get_by_role("button", name=re.compile(r"csv", re.I)),
            ]:
                try:
                    if loc.count():
                        export_trigger = loc.first
                        break
                except Exception:
                    continue

            if export_trigger is None:
                dump_debug("export CSV control missing")

            download_holder: dict[str, object] = {}

            def on_download(download) -> None:  # pragma: no cover - Playwright callback
                if "download" not in download_holder:
                    download_holder["download"] = download

            page.on("download", on_download)
            try:
                export_trigger.click()

                # Wait for either a direct download, a download dialog, or an email-export prompt.
                deadline = time.time() + 180
                auth_dialog = page.locator(".ui-dialog[aria-describedby='dialog-confirmation']")
                dialog = page.locator(".ui-dialog[aria-describedby='download-dialog']")
                while time.time() < deadline:
                    raise_if_internal_error("portal internal server error during export")
                    if "download" in download_holder:
                        download = download_holder["download"]
                        download.save_as(output_path)  # type: ignore[attr-defined]
                        close_modal_if_present()
                        return True
                    try:
                        if auth_dialog.count() and auth_dialog.first.is_visible():
                            close_modal_if_present()
                            raise ExportRequiresPersonalData
                    except ExportRequiresPersonalData:
                        raise
                    except Exception:
                        pass
                    try:
                        if dialog.count() and dialog.first.is_visible():
                            break
                    except Exception:
                        pass
                    page.wait_for_timeout(250)
            finally:
                try:
                    page.remove_listener("download", on_download)
                except Exception:
                    pass

            raise_if_internal_error("portal internal server error during export")

            dialog = page.locator(".ui-dialog[aria-describedby='download-dialog']")
            try:
                dialog.wait_for(state="visible", timeout=60000)
            except PlaywrightTimeoutError:
                dump_debug("export flow did not open download dialog")

            # Wait until the file is prepared (caption becomes visible and contains "ready"),
            # then click the dialog's download button to trigger an actual browser download.
            caption = dialog.locator(".download-progressbar-caption")
            try:
                caption.wait_for(state="visible", timeout=180000)
                try:
                    page.wait_for_function(
                        "(el) => el && /ready/i.test(el.textContent || '')",
                        caption.element_handle(),
                        timeout=180000,
                    )
                except Exception:
                    pass
            except PlaywrightTimeoutError:
                pass

            download_button = dialog.get_by_role("button", name=re.compile(r"download", re.I))
            try:
                download_button.wait_for(state="visible", timeout=60000)
            except PlaywrightTimeoutError:
                dump_debug("download button missing in export dialog")

            try:
                with page.expect_download(timeout=180000) as download_info:
                    download_button.click()
                download_info.value.save_as(output_path)
                close_modal_if_present()
                return True
            except PlaywrightTimeoutError:
                dump_debug("export dialog download did not start")

        try:
            def ensure_search_form() -> None:
                page.goto(TAM_URL, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                raise_if_internal_error("portal internal server error while loading landing page")
                accept_cookies_if_present()
                set_granting_country()
                # If we're on the landing page, we need to submit once to reach the search form.
                if not has_date_inputs():
                    click_search(wait_for_navigation=True)
                    accept_cookies_if_present()
                if not has_date_inputs():
                    dump_debug("date range inputs missing")

            for attempt in range(1, 6):
                try:
                    ensure_search_form()
                    break
                except TransientPortalError:
                    if attempt >= 5:
                        dump_debug("portal internal server error persisted while loading search form")
                    reset_session()
                    time.sleep(min(300, 10 * (2 ** (attempt - 1))))

            tmp_destination = destination.with_suffix(destination.suffix + ".partial")
            progress_path = parts_dir / f"{year}_done_days.txt"

            def load_done_days() -> set[date]:
                if not progress_path.exists():
                    return set()
                try:
                    content = progress_path.read_text(encoding="utf-8")
                except Exception:
                    return set()
                done: set[date] = set()
                for line in content.splitlines():
                    candidate = line.strip()
                    if not candidate:
                        continue
                    try:
                        done.add(date.fromisoformat(candidate))
                    except ValueError:
                        continue
                return done

            done_days = load_done_days()
            expected_days = (date(year, 12, 31) - date(year, 1, 1)).days + 1

            def save_done_days() -> None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
                lines = "\n".join(sorted(day.isoformat() for day in done_days))
                tmp_path.write_text(lines + ("\n" if lines else ""), encoding="utf-8")
                tmp_path.replace(progress_path)

            def mark_done(start: date, end: date) -> None:
                current = start
                while current <= end:
                    done_days.add(current)
                    current += timedelta(days=1)
                save_done_days()

            def range_is_done(start: date, end: date) -> bool:
                current = start
                while current <= end:
                    if current not in done_days:
                        return False
                    current += timedelta(days=1)
                return True

            resume = tmp_destination.exists() and progress_path.exists()
            if not resume:
                for path in [tmp_destination, destination, progress_path]:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                done_days = set()
            wrote_any = tmp_destination.exists() and tmp_destination.stat().st_size > 0

            def append_part(part_path: Path) -> None:
                nonlocal wrote_any
                if not part_path.exists():
                    return
                if not wrote_any:
                    part_path.replace(tmp_destination)
                    wrote_any = True
                    return
                with tmp_destination.open("ab") as dst, part_path.open("rb") as src:
                    src.readline()  # header
                    dst.write(src.read())
                part_path.unlink(missing_ok=True)

            def export_range(start: date, end: date) -> None:
                if start > end:
                    return
                part_path = parts_dir / f"{year}_{start.isoformat()}_{end.isoformat()}.csv"
                part_path.unlink(missing_ok=True)
                try:
                    go_back_to_search_form()
                    fill_date_range_for(start, end)
                    click_search(wait_for_navigation=True)
                    results_deadline = time.time() + 60
                    while time.time() < results_deadline:
                        raise_if_internal_error("portal internal server error while loading results")
                        if page.locator("#resultsTable").count():
                            break
                        if (
                            page.locator("a.exportCSVLink[href*='format=CSV' i]").count()
                            or page.locator("button#exportCsvButton").count()
                            or page.locator("button#newUserCriteriaToExportCsv").count()
                        ):
                            break
                        page.wait_for_timeout(250)
                    if not (
                        page.locator("#resultsTable").count()
                        or page.locator("a.exportCSVLink[href*='format=CSV' i]").count()
                        or page.locator("button#exportCsvButton").count()
                        or page.locator("button#newUserCriteriaToExportCsv").count()
                    ):
                        dump_debug("results page did not load")
                    exported = export_csv(part_path)
                    if exported:
                        append_part(part_path)
                    else:
                        mark_done(start, end)
                        return
                    mark_done(start, end)
                finally:
                    go_back_to_search_form()

            def export_range_split(start: date, end: date) -> None:
                if range_is_done(start, end):
                    return
                max_attempts = 8
                min_internal_split_days = 14
                for attempt in range(1, max_attempts + 1):
                    try:
                        export_range(start, end)
                        return
                    except ExportRequiresPersonalData:
                        if start >= end:
                            dump_debug("export requires personal data even for a single-day range")
                        mid = start + timedelta(days=(end - start).days // 2)
                        export_range_split(start, mid)
                        export_range_split(mid + timedelta(days=1), end)
                        return
                    except TransientPortalError:
                        if attempt < max_attempts:
                            reset_session()
                            time.sleep(min(300, 10 * (2 ** (attempt - 1))))
                            continue
                        if start >= end:
                            dump_debug(
                                f"portal internal server error persisted for single day {start.isoformat()}"
                            )
                        if (end - start).days < min_internal_split_days:
                            dump_debug(
                                "portal internal server error persisted for range "
                                f"{start.isoformat()}..{end.isoformat()}"
                            )
                        mid = start + timedelta(days=(end - start).days // 2)
                        export_range_split(start, mid)
                        export_range_split(mid + timedelta(days=1), end)
                        return

            # Start with the full year; split if the portal requires email-based export.
            export_range_split(date(year, 1, 1), date(year, 12, 31))
            if len(done_days) == expected_days:
                try:
                    progress_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if wrote_any and tmp_destination.exists() and len(done_days) == expected_days:
                tmp_destination.replace(destination)
        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()


def download_stateaid(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    countries_path = paths.resolve_project_path(config.countries.include_csv)
    countries = _load_countries(countries_path)

    if sample:
        logger.info("Creating sample State aid CSVs")
        sample_countries = countries[:2]
        sample_end = min(config.years.end, config.years.start + 3)
        sample_years = list(range(config.years.start, sample_end + 1))
        for row in sample_countries:
            iso3 = row["iso3"]
            for year in sample_years:
                dest = paths.stateaid_raw_dir(iso3) / f"{year}.csv"
                if dest.exists() and not force:
                    continue
                data = [
                    {
                        "Member state": iso3,
                        "Granting date": f"{year}-06-30",
                        "Beneficiary": "Sample Co A",
                        "Aid element (EUR)": 100000.0,
                        "Aid instrument": "Grant",
                        "Aid objective": "Regional development",
                        "NACE code": "24.10",
                    },
                    {
                        "Member state": iso3,
                        "Granting date": f"{year}-09-30",
                        "Beneficiary": "Sample Co B",
                        "Aid element (EUR)": 50000.0,
                        "Aid instrument": "Grant",
                        "Aid objective": "Regional development",
                        "NACE code": "24.10",
                    }
                ]
                pd.DataFrame(data).to_csv(dest, index=False)
        record_manifest(
            paths,
            config.model_dump(),
            "stateaid_download",
            [],
            [paths.data_raw / "stateaid"],
        )
        return

    if config.stateaid.backend == "eusa":
        destination = paths.data_raw / "stateaid" / "eusa_awards.csv"
        if destination.exists() and not force:
            logger.info("EUSA awards already downloaded")
        else:
            _download_eusa(destination)
        record_manifest(
            paths,
            config.model_dump(),
            "stateaid_download",
            [],
            [destination],
        )
        return

    years = range(config.years.start, config.years.end + 1)
    for row in countries:
        iso3 = row["iso3"]
        name = row.get("name", "")
        if iso3 in config.countries.exclude_iso3:
            continue
        for year in years:
            destination = paths.stateaid_raw_dir(iso3) / f"{year}.csv"
            partial = destination.with_suffix(destination.suffix + ".partial")
            if destination.exists() and not force and not partial.exists():
                logger.info("Skipping existing %s", destination)
                continue
            logger.info("Downloading %s %s", iso3, year)
            _download_playwright(iso3, name, year, destination, config.stateaid.headless)
            time.sleep(1)
    record_manifest(
        paths,
        config.model_dump(),
        "stateaid_download",
        [],
        [paths.data_raw / "stateaid"],
    )


def request_stateaid_email_exports(
    config: AppConfig,
    *,
    first_name: str,
    last_name: str,
    email: str,
    force: bool = False,
    select_all_countries: bool = True,
    years: Iterable[int] | None = None,
    headless: bool | None = None,
) -> None:
    """Submit email-based export requests on the State Aid portal.

    This does NOT download any CSVs. It drives the portal's "export by email" flow
    (personal data authorization + user details form) and records which date ranges
    have been requested so the command can be resumed without re-submitting.
    """

    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    raw_dir = paths.data_raw / "stateaid"
    requests_dir = raw_dir / "_email_requests"
    debug_dir = requests_dir / "_debug"
    requests_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    progress_path = requests_dir / "requests.csv"

    def progress_key(start: date, end: date) -> str:
        return f"{start.isoformat()}..{end.isoformat()}"

    def load_progress() -> dict[str, str]:
        if force or not progress_path.exists():
            return {}
        progress: dict[str, str] = {}
        try:
            with progress_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    key = row.get("range") or ""
                    status = row.get("status") or ""
                    if key and status:
                        progress[key] = status
        except Exception:
            return {}
        return progress

    def append_progress(start: date, end: date, status: str) -> None:
        is_new = not progress_path.exists()
        with progress_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["range", "status"])
            if is_new:
                writer.writeheader()
            writer.writerow({"range": progress_key(start, end), "status": status})

    progress = load_progress()

    if years is None:
        years_iter = range(config.years.start, config.years.end + 1)
    else:
        years_iter = list(years)

    # Country selection comes from config unless explicitly selecting all.
    countries_path = paths.resolve_project_path(config.countries.include_csv)
    countries = _load_countries(countries_path)
    included_iso3 = [
        row["iso3"]
        for row in countries
        if row.get("iso3") and row["iso3"] not in config.countries.exclude_iso3
    ]

    def last_day_of_month(year: int, month: int) -> int:
        return calendar.monthrange(year, month)[1]

    def add_months(d: date, months: int) -> date:
        year = d.year + (d.month - 1 + months) // 12
        month = (d.month - 1 + months) % 12 + 1
        day = min(d.day, last_day_of_month(year, month))
        return date(year, month, day)

    def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
        ranges: list[tuple[date, date]] = []
        current = date(start.year, start.month, 1)
        while current <= end:
            m_end = date(current.year, current.month, last_day_of_month(current.year, current.month))
            ranges.append((max(start, current), min(end, m_end)))
            current = add_months(current, 1)
        return ranges

    def quarter_ranges(start: date, end: date) -> list[tuple[date, date]]:
        # Assumes start/end are aligned to quarter boundaries when used by the splitter.
        ranges: list[tuple[date, date]] = []
        current = date(start.year, start.month, 1)
        while current <= end:
            q_end_month = add_months(current, 2)
            q_end = date(q_end_month.year, q_end_month.month, last_day_of_month(q_end_month.year, q_end_month.month))
            ranges.append((max(start, current), min(end, q_end)))
            current = add_months(current, 3)
        return ranges

    def split_range(start: date, end: date, granularity: str) -> list[tuple[date, date]]:
        if granularity == "half":
            mid = date(start.year, 6, 30)
            if start <= mid < end:
                return [(start, mid), (mid + timedelta(days=1), end)]
            return month_ranges(start, end)
        if granularity == "quarter":
            return quarter_ranges(start, end)
        if granularity == "month":
            return month_ranges(start, end)
        return []

    with sync_playwright() as p:
        try:
            launch_headless = config.stateaid.headless if headless is None else headless
            browser = p.chromium.launch(headless=launch_headless)
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message:
                raise RuntimeError(
                    "Playwright Chromium executable is missing. Run `uv run playwright install chromium` "
                    "(or `uv run playwright install`) and re-run."
                ) from exc
            raise

        context = browser.new_context()
        page = context.new_page()
        pii_entered = False

        def reset_session() -> None:
            nonlocal context, page, pii_entered
            try:
                context.close()
            except Exception:
                pass
            context = browser.new_context()
            page = context.new_page()
            pii_entered = False

        def write_debug(prefix: str) -> None:
            nonlocal pii_entered
            if not pii_entered:
                try:
                    page.screenshot(path=str(debug_dir / f"{prefix}.png"), full_page=True)
                except Exception:
                    pass
            try:
                content = page.content()
                for needle, repl in [
                    (email, "[REDACTED_EMAIL]"),
                    (first_name, "[REDACTED_FIRST_NAME]"),
                    (last_name, "[REDACTED_LAST_NAME]"),
                ]:
                    if needle:
                        content = content.replace(needle, repl)
                (debug_dir / f"{prefix}.html").write_text(content, encoding="utf-8")
            except Exception:
                pass

        def is_internal_error_page() -> bool:
            try:
                title = page.title()
                if title and re.search(r"internal application error", title, re.I):
                    return True
            except Exception:
                pass
            try:
                errors = page.locator("div.errors")
                if errors.count():
                    text = errors.first.inner_text()
                    if re.search(r"internal server error", text, re.I):
                        return True
            except Exception:
                pass
            return False

        def raise_if_internal_error(context_msg: str, prefix: str) -> None:
            if not is_internal_error_page():
                return
            write_debug(prefix)
            raise TransientPortalError(context_msg)

        def click_if_visible(locator) -> bool:
            try:
                if locator.count() and locator.first.is_visible():
                    locator.first.click()
                    return True
            except Exception:
                return False
            return False

        def accept_cookies_if_present() -> None:
            for loc in [
                page.get_by_role("button", name=re.compile(r"accept all", re.I)),
                page.get_by_role("button", name=re.compile(r"accept", re.I)),
                page.get_by_role("button", name=re.compile(r"agree", re.I)),
                page.get_by_role("button", name=re.compile(r"ok", re.I)),
            ]:
                if click_if_visible(loc):
                    page.wait_for_timeout(500)
                    return

        def has_search_form() -> bool:
            return (
                page.locator(
                    "input#dateGrantedFrom, input[name='dateGrantedFrom'], input[id*='dateGrantedFrom' i]"
                ).count()
                > 0
            )

        def ensure_search_form(prefix: str) -> None:
            page.goto(TAM_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            raise_if_internal_error("portal internal server error while loading landing page", prefix)
            accept_cookies_if_present()

            if select_all_countries:
                select_all = page.locator("input#selectAll")
                if select_all.count():
                    try:
                        select_all.first.check()
                    except Exception:
                        try:
                            select_all.first.click()
                        except Exception:
                            pass
            else:
                for iso3 in included_iso3:
                    checkbox = page.locator(
                        f"form#publicSearchForm input[type='checkbox'][name='countries'][value='Country{iso3}']"
                    )
                    if not checkbox.count():
                        continue
                    try:
                        checkbox.first.check()
                    except Exception:
                        pass

            # Submit once to reach the search form.
            search_button = page.get_by_role("button", name=re.compile(r"^search$", re.I))
            try:
                with page.expect_navigation(timeout=60000):
                    search_button.first.click()
            except PlaywrightTimeoutError:
                # Some portal flows might not trigger a full navigation.
                try:
                    search_button.first.click()
                except Exception:
                    pass
            page.wait_for_timeout(1500)
            raise_if_internal_error("portal internal server error while opening search form", prefix)

            if not has_search_form():
                # Fallback: sometimes the search form is already open.
                if page.url.endswith("/public/search") or "/public/search" in page.url:
                    pass
                else:
                    write_debug(prefix)
                    raise RuntimeError(f"Could not reach /public/search; debug saved to {debug_dir}.")

        def go_back_to_search_form(prefix: str) -> None:
            if has_search_form():
                return
            raise_if_internal_error("portal internal server error while navigating back", prefix)
            for loc in [
                page.locator("a.button-back"),
                page.get_by_role("link", name=re.compile(r"^back$", re.I)),
                page.get_by_role("button", name=re.compile(r"^back$", re.I)),
            ]:
                try:
                    if not loc.count():
                        continue
                    try:
                        with page.expect_navigation(timeout=60000):
                            loc.first.click()
                    except PlaywrightTimeoutError:
                        try:
                            loc.first.click()
                        except Exception:
                            pass
                    page.wait_for_timeout(1000)
                    if has_search_form():
                        return
                except Exception:
                    continue
            # Last resort: rebuild session.
            reset_session()
            ensure_search_form(prefix)

        def fill_date_range(start: date, end: date, prefix: str) -> None:
            start_str = start.strftime("%d/%m/%Y")
            end_str = end.strftime("%d/%m/%Y")

            def fill_first(selectors: list[str], value: str) -> bool:
                for sel in selectors:
                    loc = page.locator(sel)
                    try:
                        if loc.count() and loc.first.is_visible():
                            loc.first.fill(value)
                            return True
                    except Exception:
                        continue
                return False

            ok_start = fill_first(
                [
                    "input#dateGrantedFrom",
                    "input[name='dateGrantedFrom']",
                    "input[id*='dateGrantedFrom' i]",
                ],
                start_str,
            )
            ok_end = fill_first(
                [
                    "input#dateGrantedTo",
                    "input[name='dateGrantedTo']",
                    "input[id*='dateGrantedTo' i]",
                ],
                end_str,
            )
            if not (ok_start and ok_end):
                write_debug(prefix)
                raise RuntimeError(f"Could not locate date inputs; debug saved to {debug_dir}.")

        def click_search(prefix: str) -> None:
            for loc in [
                page.get_by_role("button", name=re.compile(r"^search$", re.I)),
                page.locator("button.button-search"),
                page.locator("button:has-text('Search')"),
            ]:
                try:
                    if loc.count():
                        try:
                            with page.expect_navigation(timeout=60000):
                                loc.first.click()
                        except PlaywrightTimeoutError:
                            try:
                                loc.first.click()
                            except Exception:
                                pass
                        page.wait_for_timeout(1500)
                        raise_if_internal_error("portal internal server error after submitting search", prefix)
                        return
                except Exception:
                    continue
            write_debug(prefix)
            raise RuntimeError(f"Could not locate Search button; debug saved to {debug_dir}.")

        def wait_for_results(prefix: str) -> None:
            deadline = time.time() + 60
            while time.time() < deadline:
                raise_if_internal_error("portal internal server error while loading results", prefix)
                if page.locator("#resultsTable").count():
                    return
                if page.locator("button#newUserCriteriaToExportCsv").count():
                    return
                if page.locator("a.exportCSVLink[href*='format=CSV' i]").count():
                    return
                page.wait_for_timeout(250)
            write_debug(prefix)
            raise RuntimeError(f"Results did not load; debug saved to {debug_dir}.")

        def request_email_export(prefix: str) -> None:
            nonlocal pii_entered
            export_button = page.locator("button#newUserCriteriaToExportCsv")
            if not export_button.count():
                if page.locator("a.exportCSVLink[href*='format=CSV' i]").count():
                    raise EmailExportNotAvailable("Direct export available (email export button not present).")
                raise EmailExportNotAvailable("Email export button not present.")

            export_button.first.click()
            page.wait_for_timeout(750)

            # Personal data authorization dialog.
            auth_dialog = page.locator(".ui-dialog[aria-describedby='dialog-confirmation']")
            try:
                if auth_dialog.count() and auth_dialog.first.is_visible():
                    ok = auth_dialog.locator("button:has-text('OK')")
                    if ok.count():
                        ok.first.click()
                        page.wait_for_timeout(750)
            except Exception:
                pass

            # Wait for personal data form (dialog or page).
            deadline = time.time() + 60
            while time.time() < deadline:
                raise_if_internal_error("portal internal server error while opening user details form", prefix)
                if page.locator(
                    "input[type='email'], input[id*='mail' i], input[name*='mail' i], input[placeholder*='mail' i]"
                ).count():
                    break
                page.wait_for_timeout(250)

            def fill_input(candidates: list[str], value: str) -> None:
                for sel in candidates:
                    loc = page.locator(sel)
                    for idx in range(min(loc.count(), 5)):
                        target = loc.nth(idx)
                        try:
                            if target.is_visible() and target.is_enabled():
                                target.fill(value)
                                return
                        except Exception:
                            continue
                raise RuntimeError("missing input")

            pii_entered = True
            try:
                fill_input(
                    [
                        "input[type='text'][id*='first' i]",
                        "input[type='text'][name*='first' i]",
                        "input[id*='first' i]",
                        "input[name*='first' i]",
                        "input[placeholder*='first' i]",
                    ],
                    first_name,
                )
                fill_input(
                    [
                        "input[type='text'][id*='last' i]",
                        "input[type='text'][name*='last' i]",
                        "input[id*='last' i]",
                        "input[name*='last' i]",
                        "input[placeholder*='last' i]",
                    ],
                    last_name,
                )
                fill_input(
                    [
                        "input[type='email']",
                        "input[id*='mail' i]",
                        "input[name*='mail' i]",
                        "input[placeholder*='mail' i]",
                    ],
                    email,
                )
            except Exception:
                write_debug(prefix)
                raise RuntimeError(f"Could not fill personal data fields; debug saved to {debug_dir}.")

            # Save icon/button.
            save_clicked = False
            for loc in [
                page.get_by_role("button", name=re.compile(r"save", re.I)),
                page.locator("button:has-text('Save')"),
                page.locator("a:has-text('Save')"),
                page.locator("button[title*='save' i], a[title*='save' i]"),
                page.locator("button:has(.ui-icon-disk), a:has(.ui-icon-disk)"),
                page.locator("button:has(.ui-icon-save), a:has(.ui-icon-save)"),
                page.locator("button[id*='save' i], a[id*='save' i]"),
            ]:
                try:
                    if loc.count() and loc.first.is_visible():
                        loc.first.click()
                        save_clicked = True
                        break
                except Exception:
                    continue
            if not save_clicked:
                write_debug(prefix)
                raise RuntimeError(f"Could not locate Save control; debug saved to {debug_dir}.")

            # Confirmation dialog: "Your request has been saved successfully" / "will be sent".
            deadline = time.time() + 60
            confirmation = None
            while time.time() < deadline:
                raise_if_internal_error("portal internal server error after saving request", prefix)
                for candidate in [
                    page.locator(".ui-dialog:has-text('saved successfully')"),
                    page.locator(".ui-dialog:has-text('will be sent')"),
                    page.locator("div#dialog-confirmation-export"),
                ]:
                    try:
                        if candidate.count() and candidate.first.is_visible():
                            confirmation = candidate.first
                            break
                    except Exception:
                        continue
                if confirmation is not None:
                    break
                page.wait_for_timeout(250)

            if confirmation is not None:
                try:
                    ok = page.locator(".ui-dialog button:has-text('OK')")
                    if ok.count():
                        ok.first.click()
                        page.wait_for_timeout(500)
                except Exception:
                    pass

        def run_range(start: date, end: date, prefix: str) -> str:
            key = progress_key(start, end)
            if key in progress and not force:
                logger.info("Skipping already requested %s (%s)", key, progress[key])
                return progress[key]
            for attempt in range(1, 5):
                try:
                    if not has_search_form():
                        ensure_search_form(prefix)
                    fill_date_range(start, end, prefix)
                    click_search(prefix)
                    wait_for_results(prefix)
                    try:
                        if (
                            page.locator("#resultsTable").count()
                            and page.locator("#resultsTable tbody tr").count() == 0
                        ):
                            append_progress(start, end, "no_results")
                            progress[key] = "no_results"
                            logger.info("No results for %s", key)
                            return "no_results"
                    except Exception:
                        pass
                    request_email_export(prefix)
                    append_progress(start, end, "submitted")
                    progress[key] = "submitted"
                    logger.info("Requested email export for %s", key)
                    return "submitted"
                except EmailExportNotAvailable as exc:
                    logger.warning("%s -> %s", key, exc)
                    append_progress(start, end, "direct_only")
                    progress[key] = "direct_only"
                    return "direct_only"
                except TransientPortalError:
                    if attempt >= 4:
                        raise
                    reset_session()
                    time.sleep(min(120, 5 * (2 ** (attempt - 1))))
                finally:
                    try:
                        go_back_to_search_form(prefix)
                    except Exception:
                        reset_session()
            return "failed"

        def run_range_split(start: date, end: date, granularity: str, prefix: str) -> None:
            key = progress_key(start, end)
            if key in progress and not force:
                return
            try:
                run_range(start, end, prefix)
                return
            except TransientPortalError:
                # Split the range on repeated 500s.
                if granularity == "month":
                    raise
                next_granularity = {"year": "half", "half": "quarter", "quarter": "month"}[granularity]
                for sub_start, sub_end in split_range(start, end, next_granularity):
                    run_range_split(sub_start, sub_end, next_granularity, prefix)

        try:
            for year in years_iter:
                year_start = date(year, 1, 1)
                year_end = date(year, 12, 31)
                prefix = f"email_{year}"
                if progress_key(year_start, year_end) in progress and not force:
                    continue
                logger.info("Requesting email export for %s (%s)", year, "all countries" if select_all_countries else "config countries")
                run_range_split(year_start, year_end, "year", prefix)
        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()
