from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class ColumnContract:
    name: str
    dtype: str
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unique: bool = False
    critical: bool = False
    max_null_ratio: Optional[float] = None


@dataclass(frozen=True)
class DataContract:
    source_name: str
    table_name: str
    columns: Sequence[ColumnContract]


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # ERROR | WARN
    code: str
    message: str


def validate_dataframe_contract(df: pd.DataFrame, contract: DataContract) -> Tuple[bool, List[ValidationIssue]]:
    issues: List[ValidationIssue] = []
    col_map: Dict[str, ColumnContract] = {c.name: c for c in contract.columns}

    for col_name, cc in col_map.items():
        if col_name not in df.columns:
            issues.append(
                ValidationIssue(
                    level="ERROR" if cc.critical else "WARN",
                    code="missing_column",
                    message=f"Missing column: {col_name}",
                )
            )
            continue

        series = df[col_name]

        null_ratio = float(series.isna().mean()) if len(series) else 0.0
        if cc.max_null_ratio is not None and null_ratio > cc.max_null_ratio:
            issues.append(
                ValidationIssue(
                    level="ERROR" if cc.critical else "WARN",
                    code="null_ratio_exceeded",
                    message=f"Column {col_name} null ratio {null_ratio:.4f} exceeds {cc.max_null_ratio:.4f}",
                )
            )

        if not cc.nullable and series.isna().any():
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    code="non_nullable_has_null",
                    message=f"Column {col_name} is non-nullable but contains nulls",
                )
            )

        if cc.unique and series.dropna().duplicated().any():
            issues.append(
                ValidationIssue(
                    level="ERROR",
                    code="unique_violation",
                    message=f"Column {col_name} has duplicate values",
                )
            )

        numeric = pd.to_numeric(series, errors="coerce")
        if cc.min_value is not None:
            bad_min = numeric.notna() & (numeric < cc.min_value)
            if bad_min.any():
                issues.append(
                    ValidationIssue(
                        level="ERROR" if cc.critical else "WARN",
                        code="min_value_violation",
                        message=f"Column {col_name} has values below {cc.min_value}",
                    )
                )

        if cc.max_value is not None:
            bad_max = numeric.notna() & (numeric > cc.max_value)
            if bad_max.any():
                issues.append(
                    ValidationIssue(
                        level="ERROR" if cc.critical else "WARN",
                        code="max_value_violation",
                        message=f"Column {col_name} has values above {cc.max_value}",
                    )
                )

    has_error = any(i.level == "ERROR" for i in issues)
    return (not has_error), issues
