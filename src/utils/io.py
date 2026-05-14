"""IO helpers: project root resolution, YAML config loading, parquet IO."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    """Return absolute path to the project root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Path may be relative to project root."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to project root if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read parquet, resolving relative paths against project root."""
    return pd.read_parquet(resolve_path(path), **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write parquet, creating parent dirs as needed."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, **kwargs)