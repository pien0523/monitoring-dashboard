"""Tests for predictive maintenance model and recommendations."""

import pandas as pd
import pytest

from src.config import SimulationConfig
from src.maintenance import (
    calculate_health_score,
    create_rolling_features,
    generate_machine_health_summary,
    predict_failure_probability,
    train_maintenance_model,
)
from src.recommendations import (
    assign_recommendation_text,
    calculate_expected_savings,
    generate_maintenance_recommendations,
)
from src.simulator import generate_synthetic_data


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    config = SimulationConfig(n_machines=7, n_days=3, random_seed=42)
    return generate_synthetic_data(config)


@pytest.fixture(scope="module")
def selected_time(sample_df: pd.DataFrame) -> pd.Timestamp:
    return sample_df["timestamp"].max()


def test_health_score_in_range() -> None:
    """Health score must be clamped to [0, 100]."""
    row = pd.Series(
        {
            "failure_probability_7d": 0.85,
            "avg_vibration_6h": 20.0,
            "avg_temperature_6h": 88.0,
            "defect_rate_6h": 0.10,
            "downtime_minutes_6h": 90.0,
            "anomaly_count_6h": 15,
        }
    )
    score = calculate_health_score(row)
    assert 0.0 <= score <= 100.0


def test_health_score_healthy_machine() -> None:
    """A healthy machine profile should yield a high health score."""
    row = pd.Series(
        {
            "failure_probability_7d": 0.05,
            "avg_vibration_6h": 10.5,
            "avg_temperature_6h": 65.0,
            "defect_rate_6h": 0.01,
            "downtime_minutes_6h": 0.0,
            "anomaly_count_6h": 0,
        }
    )
    score = calculate_health_score(row)
    assert score >= 70.0


def test_create_rolling_features_returns_expected_columns(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """create_rolling_features must return at least machine_id and core feature cols."""
    result = create_rolling_features(sample_df, selected_time)
    assert "machine_id" in result.columns
    assert "avg_temperature_6h" in result.columns
    assert "avg_vibration_6h" in result.columns
    assert "latent_degradation" in result.columns


def test_create_rolling_features_one_row_per_machine(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """create_rolling_features must return one row per machine."""
    result = create_rolling_features(sample_df, selected_time)
    assert len(result) == sample_df["machine_id"].nunique()


def test_predict_failure_probability_with_model(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """Failure probabilities must be in [0, 1] when model is available."""
    model = train_maintenance_model(sample_df)
    features = create_rolling_features(sample_df, selected_time)
    result = predict_failure_probability(features, model)
    assert "failure_probability_7d" in result.columns
    assert result["failure_probability_7d"].between(0.0, 1.0).all()


def test_predict_failure_probability_fallback(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """Fallback rule-based probability must still return [0, 1] values."""
    features = create_rolling_features(sample_df, selected_time)
    result = predict_failure_probability(features, model=None)
    assert "failure_probability_7d" in result.columns
    assert result["failure_probability_7d"].between(0.0, 1.0).all()


def test_generate_machine_health_summary_not_empty(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """Health summary must return a non-empty DataFrame."""
    result = generate_machine_health_summary(sample_df, selected_time)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_maintenance_recommendation_columns(
    sample_df: pd.DataFrame, selected_time: pd.Timestamp
) -> None:
    """generate_maintenance_recommendations must include key columns."""
    summary = generate_machine_health_summary(sample_df, selected_time)
    result = generate_maintenance_recommendations(summary)
    for col in ["machine_id", "health_score", "failure_probability_7d", "recommendation", "expected_savings"]:
        assert col in result.columns, f"Missing column: {col}"


def test_expected_savings_is_numeric() -> None:
    """calculate_expected_savings must return a float."""
    result = calculate_expected_savings(0.80, planned_cost=1500, unplanned_cost=8000)
    assert isinstance(result, float)
    assert result == pytest.approx(4900.0)


def test_assign_recommendation_high_risk() -> None:
    assert assign_recommendation_text(0.85) == "Schedule maintenance within 24–48 hours"


def test_assign_recommendation_medium_risk() -> None:
    assert assign_recommendation_text(0.55) == "Inspect during next planned downtime"


def test_assign_recommendation_low_risk() -> None:
    assert assign_recommendation_text(0.20) == "Continue normal monitoring"


def test_time_window_filtering(sample_df: pd.DataFrame) -> None:
    """Selecting a historical time should return fewer rows than max time."""
    t_max = sample_df["timestamp"].max()
    t_mid = sample_df["timestamp"].min() + (t_max - sample_df["timestamp"].min()) / 2
    full = sample_df[sample_df["timestamp"] <= t_max]
    partial = sample_df[sample_df["timestamp"] <= t_mid]
    assert len(partial) < len(full)
