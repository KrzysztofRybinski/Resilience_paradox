"""Configuration loader for Resilience Paradox."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class YearsConfig(BaseModel):
    start: int
    end: int
    base_start: int
    base_end: int


class CountriesConfig(BaseModel):
    include_csv: str
    exclude_iso3: List[str] = Field(default_factory=list)


class StateAidConfig(BaseModel):
    backend: str = "playwright"
    headless: bool = True
    max_rows_per_export: int = 50000


class OECDICIOConfig(BaseModel):
    release: str
    bundles: List[str]


class OECDConfig(BaseModel):
    icio: OECDICIOConfig


class PathsConfig(BaseModel):
    data_raw: str
    data_int: str
    data_final: str
    output: str


class AppConfig(BaseModel):
    years: YearsConfig
    countries: CountriesConfig
    stateaid: StateAidConfig
    oecd: OECDConfig
    paths: PathsConfig

    model_config = {
        "extra": "allow",
    }


def load_config(path: str | Path) -> AppConfig:
    """Load TOML config into AppConfig."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return AppConfig.model_validate(payload)


def load_shocks(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload
