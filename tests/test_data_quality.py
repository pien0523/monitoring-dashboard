"""Tests for data quality validation and CSV preparation."""

import pandas as pd
import pytest

from src.data_quality import (
    has_blocking_errors,
    prepare_uploaded_data,
    validate_data_quality,
)


def _valid_df(n: int = 50) -> pd.DataFrame:
    """Build a minimal valid sensor DataFrame for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
            "machine_id": [f"M-00{(i % 3) + 1}" for i in range(n)],
            "status": ["running"] * n,
            "utilization_pct": [80.0] * n,
            "temperature_c": [70.0] * n,
            "vibration_hz": [11.0] * n,
            "output_count": [2] * n,
            "defect_count": [0] * n,
            "cycle_time_sec": [24.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# prepare_uploaded_data
# ---------------------------------------------------------------------------

def test_prepare_converts_timestamp_strings() -> None:
    """prepare_uploaded_data must parse timestamp strings into datetime."""
    df = _valid_df()
    df["timestamp"] = df["timestamp"].astype(str)  # simulate CSV string column
    result = prepare_uploaded_data(df)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_prepare_converts_numeric_columns() -> None:
    """prepare_uploaded_data must coerce numeric columns from object dtype."""
    df = _valid_df()
    for col in ["utilization_pct", "temperature_c", "vibration_hz",
                "output_count", "defect_count", "cycle_time_sec"]:
        df[col] = df[col].astype(str)  # simulate CSV reading everything as string
    result = prepare_uploaded_data(df)
    assert pd.api.types.is_float_dtype(result["temperature_c"])
    assert pd.api.types.is_float_dtype(result["utilization_pct"])


def test_prepare_does_not_mutate_input() -> None:
    """prepare_uploaded_data must return a copy, not modify the original."""
    df = _valid_df()
    original_dtype = df["timestamp"].dtype
    prepare_uploaded_data(df)
    assert df["timestamp"].dtype == original_dtype


# ---------------------------------------------------------------------------
# validate_data_quality — passing case
# ---------------------------------------------------------------------------

def test_valid_dataframe_passes() -> None:
    """A clean dataset must produce no blocking errors and no warnings."""
    df = _valid_df()
    report = validate_data_quality(df)
    assert not has_blocking_errors(report)
    assert len(report["warnings"]) == 0


# ---------------------------------------------------------------------------
# validate_data_quality — blocking errors
# ---------------------------------------------------------------------------

def test_missing_required_columns_blocks() -> None:
    """Dropping a required column must produce a blocking error."""
    df = _valid_df().drop(columns=["vibration_hz"])
    report = validate_data_quality(df)
    assert has_blocking_errors(report)
    assert any("vibration_hz" in e for e in report["blocking_errors"])


def test_invalid_timestamp_blocks() -> None:
    """Unparseable timestamp values must produce a blocking error."""
    df = _valid_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Inject NaT to simulate parse failure (as prepare_uploaded_data would produce)
    df.loc[0, "timestamp"] = pd.NaT
    report = validate_data_quality(df)
    assert has_blocking_errors(report)
    assert any("timestamp" in e for e in report["blocking_errors"])


def test_invalid_status_values_blocks() -> None:
    """Status values outside the allowed set must produce a blocking error."""
    df = _valid_df()
    df.loc[0, "status"] = "broken"
    report = validate_data_quality(df)
    assert has_blocking_errors(report)
    assert any("status" in e for e in report["blocking_errors"])


# ---------------------------------------------------------------------------
# validate_data_quality — warnings
# ---------------------------------------------------------------------------

def test_duplicate_machine_timestamp_detected() -> None:
    """Duplicate (machine_id, timestamp) pairs must produce a warning."""
    df = _valid_df()
    duplicate_row = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)
    report = validate_data_quality(df)
    assert any("duplicate" in w.lower() and "timestamp" in w.lower() for w in report["warnings"])


def test_negative_numeric_values_detected() -> None:
    """Negative temperature values must produce a warning."""
    df = _valid_df()
    df.loc[0, "temperature_c"] = -5.0
    report = validate_data_quality(df)
    assert any("temperature_c" in w and "negative" in w.lower() for w in report["warnings"])


def test_defect_count_exceeds_output_count_detected() -> None:
    """Rows where defect_count > output_count must produce a warning."""
    df = _valid_df()
    df.loc[0, "defect_count"] = 10
    df.loc[0, "output_count"] = 2
    report = validate_data_quality(df)
    assert any("defect_count" in w and "output_count" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# has_blocking_errors
# ---------------------------------------------------------------------------

def test_has_blocking_errors_true() -> None:
    report = {"blocking_errors": ["some error"], "warnings": []}
    assert has_blocking_errors(report) is True


def test_has_blocking_errors_false() -> None:
    report = {"blocking_errors": [], "warnings": ["minor warning"]}
    assert has_blocking_errors(report) is False
