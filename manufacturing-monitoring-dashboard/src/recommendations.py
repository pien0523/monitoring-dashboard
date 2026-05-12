"""Maintenance recommendations and cost-benefit calculations."""

from __future__ import annotations

import pandas as pd

from src.config import MaintenanceConfig
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = MaintenanceConfig()


def calculate_expected_savings(
    failure_probability: float,
    planned_cost: float = _CFG.planned_maintenance_cost,
    unplanned_cost: float = _CFG.unplanned_failure_cost,
) -> float:
    """Compute expected savings from performing planned maintenance.

    Formula: P(failure) × unplanned_cost − planned_cost

    Args:
        failure_probability: 7-day failure probability (0–1).
        planned_cost: Cost of scheduled maintenance.
        unplanned_cost: Cost of an unplanned failure.

    Returns:
        Expected savings in currency units. Negative means maintenance is not yet justified.
    """
    expected_failure_cost = failure_probability * unplanned_cost
    return round(expected_failure_cost - planned_cost, 2)


def assign_recommendation_text(failure_probability: float) -> str:
    """Map failure probability to a human-readable maintenance recommendation.

    Args:
        failure_probability: 7-day failure probability (0–1).

    Returns:
        Recommendation string.
    """
    if failure_probability >= _CFG.high_risk_threshold:
        return "Schedule maintenance within 24–48 hours"
    if failure_probability >= _CFG.medium_risk_threshold:
        return "Inspect during next planned downtime"
    return "Continue normal monitoring"


def generate_maintenance_recommendations(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Produce a final recommendation table from the machine health summary.

    Args:
        summary_df: Output from generate_machine_health_summary with health_score
            and failure_probability_7d columns.

    Returns:
        DataFrame with columns: machine_id, health_score, failure_probability_7d,
        recommendation, confidence, expected_savings, priority.
    """
    if summary_df.empty:
        logger.warning("generate_maintenance_recommendations received empty summary.")
        return pd.DataFrame(
            columns=[
                "machine_id",
                "health_score",
                "failure_probability_7d",
                "recommendation",
                "confidence",
                "expected_savings",
                "priority",
            ]
        )

    result = summary_df.copy()

    if "failure_probability_7d" not in result.columns:
        logger.warning("failure_probability_7d column missing; defaulting to 0.")
        result["failure_probability_7d"] = 0.0

    if "health_score" not in result.columns:
        result["health_score"] = 100.0

    result["recommendation"] = result["failure_probability_7d"].apply(
        assign_recommendation_text
    )

    result["expected_savings"] = result["failure_probability_7d"].apply(
        calculate_expected_savings
    )

    # Confidence: proxy using failure_probability distance from 0.5 midpoint
    result["confidence"] = (
        (result["failure_probability_7d"] - 0.5).abs() * 2
    ).clip(0.0, 1.0).round(3)

    # Priority: 1 = most urgent
    result["priority"] = (
        result["failure_probability_7d"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    output_cols = [
        "machine_id",
        "health_score",
        "failure_probability_7d",
        "recommendation",
        "confidence",
        "expected_savings",
        "priority",
    ]
    existing = [c for c in output_cols if c in result.columns]
    return result[existing].sort_values("priority").reset_index(drop=True)
