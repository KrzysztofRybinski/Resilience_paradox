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


def _resolve_existing_bundle_zip(raw_dir: Path, release: str, bundle: str) -> Path | None:
    """Return an existing ZIP path for the bundle, even if the filename is non-standard."""
    expected = raw_dir / f"ICIO{release}_{bundle}.zip"
    if expected.exists():
        return expected
    candidates = sorted(raw_dir.glob(f"*{bundle}*.zip"))
    if candidates:
        return candidates[0]
    candidates = sorted(raw_dir.glob(f"*{bundle.replace('-', '_')}*.zip"))
    if candidates:
        return candidates[0]
    return None


def _normalize_href(href: str) -> str:
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.oecd.org" + href
    return href


def _fetch_icio_page_html(logger) -> str:
    try:
        return fetch_text(OECD_ICIO_PAGE)
    except Exception as exc:
        logger.warning(
            "Failed to fetch OECD ICIO dataset page via requests (%s). Trying Playwright.",
            exc,
        )

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional browser install
        raise RuntimeError(
            "Failed to fetch OECD ICIO dataset page, and Playwright is unavailable.\n"
            "Fix: install Playwright browsers with `uv run playwright install chromium` "
            "and retry `rp oecd download`."
        ) from exc

    try:
        with sync_playwright() as p:  # pragma: no cover - network dependent
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(OECD_ICIO_PAGE, wait_until="networkidle", timeout=90_000)
                return page.content()
            finally:
                browser.close()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to fetch OECD ICIO dataset page via requests and via Playwright.\n"
            "Manual fallback: download the bundles in a browser and place them under data/raw/oecd_icio/ "
            "with names ICIO<release>_<bundle>.zip."
        ) from exc


def _find_bundle_urls(html: str, bundles: list[str]) -> dict[str, str]:
    urls: dict[str, str] = {}

    def strip_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    # Prefer "fileview*.aspx?IDFile=..." links where the surrounding anchor text includes
    # the bundle label like "2011-2015" or "2016-2022".
    anchor_pat = re.compile(
        r'<a[^>]+href="(?P<href>[^"]*fileview\d*\.aspx\?IDFile=[^"]+)"[^>]*>(?P<body>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in anchor_pat.finditer(html):
        href = _normalize_href(m.group("href"))
        body = re.sub(r"\s+", " ", strip_tags(m.group("body"))).strip()
        for bundle in bundles:
            if bundle in urls:
                continue
            if bundle in body or bundle in href:
                urls[bundle] = href

    # Legacy fallback: sometimes direct .zip links exist in HTML
    for url in re.findall(r"https?://[^\"']+\.zip", html):
        for bundle in bundles:
            if bundle in url:
                urls[bundle] = url

    return urls


def download_icio_bundles(config: AppConfig, refresh: bool = False) -> None:
    logger = setup_logging()
    paths = Paths.from_config(config)
    paths.ensure()
    raw_dir = paths.oecd_raw_dir()

    bundles = list(config.oecd.icio.bundles)
    release = config.oecd.icio.release
    destinations = {bundle: raw_dir / f"ICIO{release}_{bundle}.zip" for bundle in bundles}

    existing_paths: dict[str, Path] = {}
    missing_paths: list[Path] = []
    for bundle, expected in destinations.items():
        existing = _resolve_existing_bundle_zip(raw_dir, release, bundle)
        if existing is None:
            missing_paths.append(expected)
        else:
            existing_paths[bundle] = existing
    # If everything is already present and we're not refreshing, do not hit the network.
    if not refresh and not missing_paths:
        logger.info("All OECD ICIO bundles already exist; skipping download.")
        record_manifest(
            paths,
            config.model_dump(),
            "oecd_download",
            [Path(OECD_ICIO_PAGE)],
            list(existing_paths.values()),
        )
        return

    logger.info("Fetching OECD ICIO dataset page for bundle URLs")
    try:
        html = _fetch_icio_page_html(logger)
    except Exception as exc:
        if missing_paths:
            missing_list = "\n".join(f"  - {path}" for path in missing_paths)
            raise RuntimeError(
                "Failed to fetch OECD ICIO dataset page (may be blocked).\n"
                "Manual fallback: place the missing ICIO ZIPs at:\n"
                f"{missing_list}"
            ) from exc
        raise

    urls = _find_bundle_urls(html, bundles)
    if not urls:
        raise RuntimeError(
            "Could not locate ICIO bundle URLs on the OECD dataset page.\n"
            "Manual fallback: download the bundles in a browser and place them under data/raw/oecd_icio/ "
            "with names ICIO<release>_<bundle>.zip."
        )

    for bundle, dest in destinations.items():
        if _resolve_existing_bundle_zip(raw_dir, release, bundle) is not None and not refresh:
            logger.info("Bundle %s already exists; skipping", dest.name)
            continue
        url = urls.get(bundle)
        if url is None:
            raise RuntimeError(f"Missing URL for bundle {bundle}. Found: {sorted(urls.keys())}")
        logger.info("Downloading OECD ICIO bundle %s", bundle)
        download_file(url, dest, headers={"Referer": OECD_ICIO_PAGE})

    record_manifest(
        paths,
        config.model_dump(),
        "oecd_download",
        [Path(OECD_ICIO_PAGE)],
        list(destinations.values()),
    )
