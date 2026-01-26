from pathlib import Path

import polars as pl

from resilience_paradox.data.oecd_icio_reader import (
    compute_import_hhi,
    compute_input_shares_base,
    compute_output_va,
    read_icio_long,
)


def test_oecd_reader_smoke():
    csv_path = Path("tests/fixtures/icio_small.csv")
    lazy = read_icio_long(csv_path, years=[2013, 2014])
    df = lazy.collect()

    output_va = compute_output_va(df)
    assert not output_va.empty
    assert set(output_va.columns) >= {"country_iso3", "icio50", "year"}

    shares = compute_input_shares_base(df, base_years=[2013, 2014])
    assert not shares.empty
    assert shares["ioshare_base"].between(0, 1).all()

    hhi = compute_import_hhi(df)
    assert "import_hhi_intermediate" in hhi.columns
