"""Download OECD ICIO bundles."""
from __future__ import annotations

import re
from pathlib import Path

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.http import download_file, fetch_text

OECD_ICIO_PAGE = "https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm"


def _find_bundle_urls(html: str) -> dict[str, str]:
    urls = {}
    pattern = re.compile(r"https?://[^\"']+\.zip")
    for match in pattern.findall(html):
        url = match
        for bundle in ["2011-2015", "2016-2022"]:
            if bundle in url:
                urls[bundle] = url
    return urls


def download_icio_bundles(config: AppConfig, force: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    raw_dir = paths.oecd_raw_dir()

    logger.info("Fetching OECD ICIO page for bundle URLs")
    html = fetch_text(OECD_ICIO_PAGE)
    urls = _find_bundle_urls(html)

    if not urls:
        raise RuntimeError(
            "Could not locate ICIO bundle URLs. Please download manually and place in data/raw/oecd_icio/."
        )

    for bundle in config.oecd.icio.bundles:
        if bundle not in urls:
            raise RuntimeError(f"Missing bundle URL for {bundle} on OECD page.")
        destination = raw_dir / f"ICIO{config.oecd.icio.release}_{bundle}.zip"
        if destination.exists() and not force:
            logger.info("Bundle %s already exists, skipping", destination.name)
            continue
        logger.info("Downloading %s", urls[bundle])
        download_file(urls[bundle], destination)
    record_manifest(
        paths,
        config.model_dump(),
        "oecd_download",
        [Path(OECD_ICIO_PAGE)],
        list(raw_dir.glob("*.zip")),
    )
