"""Tests for anomaly detection."""

import pandas as pd
import pytest

from src.anomaly import calculate_anomaly_severity, create_alert_feed, detect_anomalies
from src.config import SimulationConfig
from src.simulator import generate_synthetic_data


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    config = SimulationConfig(n_machines=3, n_days=1, random_seed=42)
    return generate_synthetic_data(config)


@pytest.fixture(scope="module")
def m003_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return sample_df[sample_df["machine_id"] == "M-003"].copy()


def test_detect_anomalies_returns_required_columns(m003_df: pd.DataFrame) -> None:
    """detect_anomalies must add anomaly_score, is_anomaly, and anomaly_severity."""
    result = detect_anomalies(m003_df, contamination=0.05)
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert "anomaly_severity" in result.columns


def test_detect_anomalies_produces_flags(m003_df: pd.DataFrame) -> None:
    """At least some rows should be flagged as anomalies in a real dataset."""
    result = detect_anomalies(m003_df, contamination=0.05)
    assert result["is_anomaly"].any()


def test_detect_anomalies_scores_in_range(m003_df: pd.DataFrame) -> None:
    """Anomaly scores must be in [0, 1]."""
    result = detect_anomalies(m003_df, contamination=0.05)
    assert result["anomaly_score"].between(0.0, 1.0).all()


def test_detect_anomalies_handles_small_dataframe() -> None:
    """detect_anomalies must return defaults without raising for small input."""
    tiny_df = pd.DataFrame(
        {
            "temperature_c": [70.0, 71.0],
            "vibration_hz": [11.0, 11.5],
            "utilization_pct": [80.0, 82.0],
            "cycle_time_sec": [24.0, 25.0],
            "machine_id": ["M-001", "M-001"],
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="min"),
        }
    )
    result = detect_anomalies(tiny_df, contamination=0.05, min_rows=30)
    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].sum() == 0


def test_anomaly_severity_high_for_critical_temp() -> None:
    """Severity must be 'high' when temperature exceeds critical threshold."""
    row = pd.Series(
        {"is_anomaly": True, "temperature_c": 87.0, "vibration_hz": 12.0, "anomaly_score": 0.9}
    )
    assert calculate_anomaly_severity(row) == "high"


def test_anomaly_severity_normal_for_non_anomaly() -> None:
    row = pd.Series({"is_anomaly": False, "temperature_c": 65.0, "vibration_hz": 10.0, "anomaly_score": 0.1})
    assert calculate_anomaly_severity(row) == "normal"


def test_create_alert_feed_returns_dataframe(sample_df: pd.DataFrame) -> None:
    """create_alert_feed must return a DataFrame with required columns."""
    selected_time = sample_df["timestamp"].max()
    result = create_alert_feed(sample_df, selected_time)
    assert isinstance(result, pd.DataFrame)
    for col in ["timestamp", "machine_id", "severity", "message"]:
        assert col in result.columns


def test_create_alert_feed_empty_input() -> None:
    """create_alert_feed must not raise on empty DataFrame."""
    result = create_alert_feed(pd.DataFrame(), pd.Timestamp("2024-01-01"))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
