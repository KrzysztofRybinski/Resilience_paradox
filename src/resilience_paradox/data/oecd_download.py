"""Download OECD ICIO bundles.

The OECD ICIO dataset page lists bundle downloads (e.g., 2011-2015, 2016-2022).
Those links often point to stats.oecd.org "fileview2.aspx?IDFile=..." endpoints rather than
direct ".zip" URLs embedded in the HTML.

This module fetches the dataset page, discovers the correct download URLs, and downloads
the bundles configured in config/default.toml.

Manual fallback remains supported: place the expected ZIP files under data/raw/oecd_icio/.
"""
from __future__ import annotations

import re
from pathlib import Path

from resilience_paradox.config import AppConfig
from resilience_paradox.logging import setup_logging
from resilience_paradox.paths import Paths
from resilience_paradox.pipeline import record_manifest
from resilience_paradox.utils.http import download_file, fetch_text

# Current dataset page (old /sti/ind/ path redirects here)
OECD_ICIO_PAGE = "https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html"


def _normalize_href(href: str) -> str:
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.oecd.org" + href
    return href


def _find_bundle_urls(html: str, bundles: list[str]) -> dict[str, str]:
    urls: dict[str, str] = {}

    # Prefer "fileview2.aspx?IDFile=..." links where the anchor text is the bundle label
    # like "2011-2015" or "2016-2022".
    anchor_pat = re.compile(
        r'href="(?P<href>[^"]*fileview2\.aspx\?IDFile=[^"]+)"[^>]*>\s*(?P<label>\d{4}-\d{4})\s*<',
        flags=re.IGNORECASE,
    )
    for m in anchor_pat.finditer(html):
        label = m.group("label").strip()
        if label in bundles and label not in urls:
            urls[label] = _normalize_href(m.group("href"))

    # Legacy fallback: sometimes direct .zip links exist in HTML
    for url in re.findall(r"https?://[^\"']+\.zip", html):
        for bundle in bundles:
            if bundle in url:
                urls[bundle] = url

    return urls


def download_icio_bundles(config: AppConfig, force: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    raw_dir = paths.oecd_raw_dir()

    bundles = list(config.oecd.icio.bundles)
    destinations = {
        bundle: raw_dir / f"ICIO{config.oecd.icio.release}_{bundle}.zip" for bundle in bundles
    }

    # If everything is already present and we're not forcing, do not hit the network.
    if not force and all(p.exists() for p in destinations.values()):
        logger.info("All OECD ICIO bundles already exist; skipping download.")
        record_manifest(
            paths,
            config.model_dump(),
            "oecd_download",
            [Path(OECD_ICIO_PAGE)],
            list(destinations.values()),
        )
        return

    logger.info("Fetching OECD ICIO dataset page for bundle URLs")
    try:
        html = fetch_text(OECD_ICIO_PAGE)
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch OECD ICIO dataset page (may be blocked).\n"
            "Manual fallback:\n"
            "  1) Open the OECD ICIO dataset page in a browser.\n"
            "  2) Download the regular ICIO bundles for the years you need (e.g., 2011-2015, 2016-2022).\n"
            "  3) Save them under data/raw/oecd_icio/ as:\n"
            f"       ICIO{config.oecd.icio.release}_2011-2015.zip\n"
            f"       ICIO{config.oecd.icio.release}_2016-2022.zip\n"
            "  4) Re-run without --force.\n"
            f"\nOriginal error: {exc}"
        ) from exc

    urls = _find_bundle_urls(html, bundles)
    if not urls:
        raise RuntimeError(
            "Could not locate ICIO bundle URLs on the OECD dataset page.\n"
            "Manual fallback: download the bundles in a browser and place them under data/raw/oecd_icio/ "
            "with names ICIO<release>_<bundle>.zip."
        )

    for bundle, dest in destinations.items():
        if dest.exists() and not force:
            logger.info("Bundle %s already exists; skipping", dest.name)
            continue
        url = urls.get(bundle)
        if url is None:
            raise RuntimeError(f"Missing URL for bundle {bundle}. Found: {sorted(urls.keys())}")
        logger.info("Downloading OECD ICIO bundle %s", bundle)
        download_file(url, dest)

    record_manifest(
        paths,
        config.model_dump(),
        "oecd_download",
        [Path(OECD_ICIO_PAGE)],
        list(destinations.values()),
    )
