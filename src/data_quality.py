"""Data quality validation for uploaded CSV files."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "status",
    "utilization_pct",
    "temperature_c",
    "vibration_hz",
    "output_count",
    "defect_count",
    "cycle_time_sec",
]

_NUMERIC_COLS = [
    "utilization_pct",
    "temperature_c",
    "vibration_hz",
    "output_count",
    "defect_count",
    "cycle_time_sec",
]

_VALID_STATUSES = {"running", "idle", "down", "maintenance"}


def prepare_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce timestamp and numeric columns to correct types.

    Applies ``pd.to_datetime`` on the timestamp column and ``pd.to_numeric``
    on all numeric sensor columns. Values that cannot be converted become NaN.

    Args:
        df: Raw DataFrame from a user-uploaded CSV.

    Returns:
        Copy of the DataFrame with corrected dtypes.
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("prepare_uploaded_data: %d rows, %d columns processed.", len(df), len(df.columns))
    return df


def validate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Run all quality checks and return a structured report.

    Checks performed:
    - Missing required columns (blocking)
    - Timestamp parse errors (blocking)
    - Missing machine_id (blocking)
    - Invalid status values (blocking)
    - Negative numeric values (warning)
    - defect_count > output_count (warning)
    - Fully duplicate rows (warning)
    - Duplicate (machine_id, timestamp) pairs (warning)
    - Missing values per required column (warning)

    Args:
        df: DataFrame that has already been through ``prepare_uploaded_data``.

    Returns:
        Dict with keys:
          ``blocking_errors`` — list of strings that must be fixed before use.
          ``warnings``        — list of strings that are notable but non-fatal.
          ``info``            — dict of summary statistics.
    """
    report: dict[str, Any] = {
        "blocking_errors": [],
        "warnings": [],
        "info": {
            "row_count": len(df),
            "column_count": len(df.columns),
        },
    }

    # 1. Missing required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report["blocking_errors"].append(
            f"Missing required columns: {missing_cols}"
        )
        logger.warning("Blocking: missing required columns %s", missing_cols)
        return report  # remaining checks are meaningless without required columns

    report["info"]["machine_count"] = int(df["machine_id"].nunique())

    # 2. Timestamp parse errors
    ts_null = int(df["timestamp"].isna().sum())
    if ts_null > 0:
        report["blocking_errors"].append(
            f"{ts_null} timestamp value(s) could not be parsed."
        )
    else:
        report["info"]["time_range"] = (
            f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → "
            f"{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
        )

    # 3. Missing machine_id
    missing_mid = int(df["machine_id"].isna().sum())
    if missing_mid > 0:
        report["blocking_errors"].append(
            f"{missing_mid} row(s) have missing machine_id."
        )

    # 4. Invalid status values
    invalid_mask = ~df["status"].isin(_VALID_STATUSES)
    n_invalid_status = int(invalid_mask.sum())
    if n_invalid_status > 0:
        bad_values = df.loc[invalid_mask, "status"].dropna().unique().tolist()
        report["blocking_errors"].append(
            f"Invalid status values found: {bad_values}. "
            f"Allowed: {sorted(_VALID_STATUSES)}."
        )

    # 5. Negative numeric values
    for col in _NUMERIC_COLS:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                report["warnings"].append(
                    f"'{col}': {n_neg} negative value(s) detected."
                )

    # 6. defect_count > output_count
    bad_defect = int((df["defect_count"] > df["output_count"]).sum())
    if bad_defect > 0:
        report["warnings"].append(
            f"{bad_defect} row(s) where defect_count exceeds output_count."
        )

    # 7. Fully duplicate rows
    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows > 0:
        report["warnings"].append(
            f"{n_dup_rows} fully duplicate row(s) found."
        )

    # 8. Duplicate (machine_id, timestamp) keys
    n_dup_keys = int(df.duplicated(subset=["machine_id", "timestamp"]).sum())
    if n_dup_keys > 0:
        report["warnings"].append(
            f"{n_dup_keys} duplicate (machine_id, timestamp) pair(s) detected."
        )

    # 9. Missing values per required column
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                report["warnings"].append(
                    f"'{col}': {n_missing} missing value(s)."
                )

    logger.info(
        "Data quality: %d blocking errors, %d warnings.",
        len(report["blocking_errors"]),
        len(report["warnings"]),
    )
    return report


def has_blocking_errors(report: dict[str, Any]) -> bool:
    """Return True if the quality report contains at least one blocking error.

    Args:
        report: Dict returned by ``validate_data_quality``.

    Returns:
        True when the dataset cannot be used safely.
    """
    return len(report.get("blocking_errors", [])) > 0
