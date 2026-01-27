from pathlib import Path

import pytest
from requests import HTTPError

from resilience_paradox.config import load_config
from resilience_paradox.data.oecd_download import download_icio_bundles
from resilience_paradox.paths import Paths


def _bundle_paths(config, root: Path) -> list[Path]:
    paths = Paths.from_config(config, root=root)
    raw_dir = paths.oecd_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    return [
        raw_dir / f"ICIO{config.oecd.icio.release}_{bundle}.zip"
        for bundle in config.oecd.icio.bundles
    ]


def test_download_icio_bundles_offline_no_network(monkeypatch, tmp_path):
    config = load_config("config/default.toml")
    bundle_paths = _bundle_paths(config, tmp_path)
    for bundle_path in bundle_paths:
        bundle_path.write_bytes(b"")

    paths = Paths.from_config(config, root=tmp_path)
    paths.ensure()

    def mock_fetch_text(*_args, **_kwargs):
        raise AssertionError("network called")

    monkeypatch.setattr("resilience_paradox.data.oecd_download.fetch_text", mock_fetch_text)
    monkeypatch.setattr("resilience_paradox.data.oecd_download.Paths.from_config", lambda cfg: paths)

    download_icio_bundles(config, refresh=False)


def test_download_icio_bundles_missing_offline_fallback(monkeypatch, tmp_path):
    config = load_config("config/default.toml")
    bundle_paths = _bundle_paths(config, tmp_path)

    paths = Paths.from_config(config, root=tmp_path)
    paths.ensure()

    def mock_fetch_text(*_args, **_kwargs):
        raise HTTPError("blocked")

    monkeypatch.setattr("resilience_paradox.data.oecd_download.fetch_text", mock_fetch_text)
    monkeypatch.setattr("resilience_paradox.data.oecd_download.Paths.from_config", lambda cfg: paths)

    with pytest.raises(RuntimeError) as excinfo:
        download_icio_bundles(config, refresh=False)

    message = str(excinfo.value)
    assert "Manual fallback" in message
    for bundle_path in bundle_paths:
        assert bundle_path.name in message
