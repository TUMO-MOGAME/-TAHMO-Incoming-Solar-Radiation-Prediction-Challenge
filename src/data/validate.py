"""Schema and range validation for competition DataFrames."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    name: str
    n_rows: int
    n_cols: int
    missing_columns: list[str]
    extra_columns: list[str]
    null_counts: dict[str, int]
    duplicate_ids: int
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_columns and self.duplicate_ids == 0 and not self.issues


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: list[str],
    id_col: str | None = None,
    range_checks: dict[str, tuple[float, float]] | None = None,
) -> ValidationReport:
    """Run schema + null + range checks. Returns a report (does not raise)."""
    cols = set(df.columns)
    required = set(required_columns)
    missing = sorted(required - cols)
    extra = sorted(cols - required)

    null_counts = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}

    dup_ids = 0
    if id_col is not None and id_col in df.columns:
        dup_ids = int(df[id_col].duplicated().sum())

    issues: list[str] = []
    if range_checks:
        for col, (lo, hi) in range_checks.items():
            if col not in df.columns:
                continue
            below = int((df[col] < lo).sum())
            above = int((df[col] > hi).sum())
            if below:
                issues.append(f"{col}: {below} values below {lo}")
            if above:
                issues.append(f"{col}: {above} values above {hi}")

    return ValidationReport(
        name=name,
        n_rows=len(df),
        n_cols=df.shape[1],
        missing_columns=missing,
        extra_columns=extra,
        null_counts=null_counts,
        duplicate_ids=dup_ids,
        issues=issues,
    )
