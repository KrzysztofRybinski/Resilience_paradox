from resilience_paradox.config import load_config


def test_load_config():
    cfg = load_config("config/default.toml")
    assert cfg.years.start == 2016
    assert cfg.countries.include_csv.endswith("countries_eu23.csv")
    assert "POL" in cfg.countries.exclude_iso3
    assert cfg.oecd.icio.release == "2025"
    assert "2011-2015" in cfg.oecd.icio.bundles
