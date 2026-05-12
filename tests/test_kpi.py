"""Tests for OEE and bottleneck calculations."""

import pandas as pd
import pytest

from src.config import OEEConfig, SimulationConfig
from src.kpi import (
    calculate_bottleneck_score,
    calculate_oee,
    calculate_oee_by_machine,
    calculate_oee_trend,
    identify_main_issue,
)
from src.simulator import generate_synthetic_data


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    config = SimulationConfig(n_machines=7, n_days=2, random_seed=42)
    return generate_synthetic_data(config)


@pytest.fixture
def oee_config() -> OEEConfig:
    return OEEConfig()


def test_oee_values_within_range(sample_df: pd.DataFrame, oee_config: OEEConfig) -> None:
    """OEE and all factor values must be in [0, 1]."""
    result = calculate_oee(sample_df, oee_config)
    for key in ["oee", "availability", "performance", "quality"]:
        val = result[key]
        assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


def test_availability_zero_for_down_machine(oee_config: OEEConfig) -> None:
    """Availability must be 0 when all rows are 'down'."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="min"),
            "machine_id": ["M-001"] * 60,
            "status": ["down"] * 60,
            "cycle_time_sec": [0.0] * 60,
            "output_count": [0] * 60,
            "defect_count": [0] * 60,
        }
    )
    result = calculate_oee(df, oee_config)
    assert result["availability"] == 0.0
    assert result["oee"] == 0.0


def test_oee_empty_dataframe(oee_config: OEEConfig) -> None:
    """calculate_oee must not raise on an empty DataFrame."""
    result = calculate_oee(pd.DataFrame(), oee_config)
    assert result["oee"] == 0.0


def test_oee_by_machine_returns_all_machines(sample_df: pd.DataFrame, oee_config: OEEConfig) -> None:
    """calculate_oee_by_machine must return one row per machine_id."""
    result = calculate_oee_by_machine(sample_df, oee_config)
    expected_machines = sorted(sample_df["machine_id"].unique())
    result_machines = sorted(result["machine_id"].unique())
    assert result_machines == expected_machines


def test_oee_trend_returns_daily_rows(sample_df: pd.DataFrame, oee_config: OEEConfig) -> None:
    """OEE trend must return rows grouped by date."""
    result = calculate_oee_trend(sample_df, oee_config)
    assert "date" in result.columns
    assert "machine_id" in result.columns
    assert "oee" in result.columns
    assert len(result) > 0


def test_identify_main_issue_high_downtime() -> None:
    """identify_main_issue must classify low availability as 'High downtime'."""
    row = pd.Series({"availability": 0.50, "performance": 0.90, "quality": 0.98, "oee": 0.44, "avg_utilization": 70})
    assert identify_main_issue(row) == "High downtime"


def test_identify_main_issue_slow_cycle() -> None:
    row = pd.Series({"availability": 0.92, "performance": 0.60, "quality": 0.98, "oee": 0.54, "avg_utilization": 70})
    assert identify_main_issue(row) == "Slow cycle time"


def test_bottleneck_score_column_exists(sample_df: pd.DataFrame, oee_config: OEEConfig) -> None:
    """calculate_bottleneck_score must add bottleneck_score and main_issue columns."""
    oee_df = calculate_oee_by_machine(sample_df, oee_config)
    result = calculate_bottleneck_score(oee_df)
    assert "bottleneck_score" in result.columns
    assert "main_issue" in result.columns
