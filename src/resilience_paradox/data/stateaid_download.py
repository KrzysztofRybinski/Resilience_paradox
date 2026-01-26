"""Download State Aid transparency data."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, wait_exponential

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest

TAM_URL = "https://webgate.ec.europa.eu/competition/transparency/public?lang=en"


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


def _download_playwright(country: str, year: int, destination: Path, headless: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(TAM_URL, wait_until="networkidle")
        page.wait_for_timeout(2000)
        # NOTE: selectors are placeholders and may need updates if the UI changes.
        page.select_option("select[name='grantingCountry']", country)
        page.fill("input[name='grantingDateFrom']", f"01/01/{year}")
        page.fill("input[name='grantingDateTo']", f"31/12/{year}")
        page.click("button#searchButton")
        page.wait_for_timeout(2000)
        with page.expect_download() as download_info:
            page.click("button#exportCsvButton")
        download = download_info.value
        download.save_as(destination)
        browser.close()


def download_stateaid(config: AppConfig, force: bool = False, sample: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()

    countries = _load_countries(paths.root / config.countries.include_csv)

    if sample:
        logger.info("Creating sample State aid CSVs")
        sample_countries = countries[:2]
        sample_years = [config.years.start, config.years.start + 1]
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
                        "Beneficiary": "Sample Co",
                        "Aid element (EUR)": 100000.0,
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
        if iso3 in config.countries.exclude_iso3:
            continue
        for year in years:
            destination = paths.stateaid_raw_dir(iso3) / f"{year}.csv"
            if destination.exists() and not force:
                logger.info("Skipping existing %s", destination)
                continue
            logger.info("Downloading %s %s", iso3, year)
            _download_playwright(iso3, year, destination, config.stateaid.headless)
            time.sleep(1)
    record_manifest(
        paths,
        config.model_dump(),
        "stateaid_download",
        [],
        [paths.data_raw / "stateaid"],
    )
