"""Path management for pipeline outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from resilience_paradox.config import AppConfig


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_int: Path
    data_final: Path
    output: Path

    @classmethod
    def from_config(cls, config: AppConfig, root: Path | None = None) -> "Paths":
        root = root or Path.cwd()
        return cls(
            root=root,
            data_raw=root / config.paths.data_raw,
            data_int=root / config.paths.data_int,
            data_final=root / config.paths.data_final,
            output=root / config.paths.output,
        )

    def ensure(self) -> None:
        for path in [self.data_raw, self.data_int, self.data_final, self.output]:
            path.mkdir(parents=True, exist_ok=True)


    @property
    def project_root(self) -> Path:
        """Return the repository root containing the `config/` directory."""
        # paths.py is at: <repo>/src/resilience_paradox/paths.py
        return Path(__file__).resolve().parents[2]

    def resolve_project_path(self, relative: str | Path) -> Path:
        """Resolve repository-relative assets even when output root is a temp dir.

        Search order:
          1) under the pipeline output root,
          2) relative to current working directory,
          3) under repository root.
        """
        rel = Path(relative)
        candidates = [self.root / rel, rel, self.project_root / rel]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[-1]

    def stateaid_raw_dir(self, iso3: str) -> Path:
        path = self.data_raw / "stateaid" / iso3
        path.mkdir(parents=True, exist_ok=True)
        return path

    def oecd_raw_dir(self) -> Path:
        path = self.data_raw / "oecd_icio"
        path.mkdir(parents=True, exist_ok=True)
        return path
