import pandas as pd

from resilience_paradox.config import load_config
from resilience_paradox.data.crosswalks import map_nace_to_icio50
from resilience_paradox.paths import Paths


def test_crosswalk_mapping_and_splits():
    config = load_config("config/default.toml")
    paths = Paths.from_config(config)
    data = pd.DataFrame(
        {
            "nace_code": ["24", "30", "24.1", "30.1"],
            "aid_amount_eur": [100.0, 200.0, 50.0, 75.0],
        }
    )
    weights = {"24": {"C24A": 0.6, "C24B": 0.4}, "30": {"C301": 0.2, "C302T309": 0.8}}
    mapped = map_nace_to_icio50(data, weights, paths)

    split_24 = mapped[mapped["nace_code"] == "24"]
    assert split_24["aid_amount_eur"].sum() == 100.0
    assert set(split_24["icio50"]) == {"C24A", "C24B"}

    mapping_241 = mapped[mapped["nace_code"] == "24.1"]["icio50"].iloc[0]
    assert mapping_241 == "C24A"
