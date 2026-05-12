"""OEE calculation and bottleneck analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import OEEConfig
from src.logger import get_logger

logger = get_logger(__name__)


def _safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator / denominator, or fallback if denominator is zero."""
    if denominator == 0 or np.isnan(denominator):
        return fallback
    return numerator / denominator


def calculate_oee(df: pd.DataFrame, config: OEEConfig) -> dict:
    """Compute overall OEE for a filtered dataframe window.

    Args:
        df: Sensor rows for the desired time window (already filtered).
        config: OEE configuration thresholds and ideal cycle time.

    Returns:
        Dict with keys: oee, availability, performance, quality,
        total_output, total_defects, running_minutes, planned_minutes.
    """
    if df.empty:
        logger.warning("calculate_oee received empty dataframe; returning zeros.")
        return {
            "oee": 0.0,
            "availability": 0.0,
            "performance": 0.0,
            "quality": 0.0,
            "total_output": 0,
            "total_defects": 0,
            "running_minutes": 0,
            "planned_minutes": 0,
        }

    planned_minutes = len(df[df["status"] != "maintenance"])
    running_minutes = len(df[df["status"] == "running"])

    availability = _safe_divide(running_minutes, planned_minutes)

    running_rows = df[(df["status"] == "running") & (df["cycle_time_sec"] > 0)]
    avg_cycle_time = running_rows["cycle_time_sec"].mean() if not running_rows.empty else 0.0
    performance = _safe_divide(config.ideal_cycle_time_sec, avg_cycle_time)
    performance = float(np.clip(performance, 0.0, config.max_performance_cap))

    total_output = int(df["output_count"].sum())
    total_defects = int(df["defect_count"].sum())
    good_units = max(total_output - total_defects, 0)
    quality = _safe_divide(good_units, total_output, fallback=1.0) if total_output > 0 else 1.0

    oee = float(np.clip(availability * performance * quality, 0.0, 1.0))

    return {
        "oee": oee,
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "total_output": total_output,
        "total_defects": total_defects,
        "running_minutes": running_minutes,
        "planned_minutes": planned_minutes,
    }


def calculate_oee_by_machine(df: pd.DataFrame, config: OEEConfig) -> pd.DataFrame:
    """Compute OEE broken down by machine_id.

    Args:
        df: Sensor DataFrame (filtered to desired window).
        config: OEE configuration.

    Returns:
        DataFrame indexed by machine_id with OEE factor columns.
    """
    if df.empty:
        logger.warning("calculate_oee_by_machine received empty dataframe.")
        return pd.DataFrame(
            columns=["machine_id", "oee", "availability", "performance", "quality"]
        )

    records = []
    for machine_id, grp in df.groupby("machine_id"):
        result = calculate_oee(grp, config)
        result["machine_id"] = machine_id
        records.append(result)

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df

    result_df["oee"] = result_df["oee"].clip(0.0, 1.0)
    return result_df.reset_index(drop=True)


def calculate_oee_trend(df: pd.DataFrame, config: OEEConfig) -> pd.DataFrame:
    """Compute daily OEE per machine for the trend window.

    Args:
        df: Sensor DataFrame covering the trend period.
        config: OEE configuration.

    Returns:
        DataFrame with columns: date, machine_id, oee, availability, performance, quality.
    """
    if df.empty:
        logger.warning("calculate_oee_trend received empty dataframe.")
        return pd.DataFrame(columns=["date", "machine_id", "oee"])

    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    records = []
    for (date, machine_id), grp in df.groupby(["date", "machine_id"]):
        result = calculate_oee(grp, config)
        result["date"] = date
        result["machine_id"] = machine_id
        records.append(result)

    if not records:
        return pd.DataFrame(columns=["date", "machine_id", "oee"])

    trend_df = pd.DataFrame(records)
    trend_df["oee"] = trend_df["oee"].clip(0.0, 1.0)
    return trend_df.sort_values(["machine_id", "date"]).reset_index(drop=True)


def identify_main_issue(row: pd.Series) -> str:
    """Classify the primary reason for low OEE for a machine row.

    Args:
        row: A Series from the OEE by-machine DataFrame.

    Returns:
        Human-readable string describing the main OEE issue.
    """
    availability = row.get("availability", 1.0)
    performance = row.get("performance", 1.0)
    quality = row.get("quality", 1.0)
    utilization = row.get("avg_utilization", 0.0)
    oee = row.get("oee", 1.0)

    if availability < 0.70:
        return "High downtime"
    if performance < 0.75:
        return "Slow cycle time"
    if quality < 0.95:
        return "High defect rate"
    if utilization > 85 and oee < 0.75:
        return "Potential bottleneck"
    return "Normal"


def calculate_bottleneck_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a bottleneck_score and main_issue column to OEE-by-machine output.

    The score combines low OEE, downtime, defect rate, and slow cycle time.
    Higher score = worse bottleneck.

    Args:
        df: DataFrame from calculate_oee_by_machine with extra columns appended.

    Returns:
        DataFrame with added bottleneck_score and main_issue columns, sorted descending.
    """
    if df.empty:
        return df

    result = df.copy()

    oee_penalty = 1.0 - result["oee"].clip(0.0, 1.0)
    avail_penalty = 1.0 - result["availability"].clip(0.0, 1.0)
    perf_penalty = 1.0 - result["performance"].clip(0.0, 1.2)
    qual_penalty = 1.0 - result["quality"].clip(0.0, 1.0)

    defect_rate = result.get("defect_rate", pd.Series(0.0, index=result.index))
    downtime_norm = result.get("downtime_minutes", pd.Series(0.0, index=result.index))
    downtime_norm = (downtime_norm / (downtime_norm.max() + 1e-9)).clip(0.0, 1.0)

    result["bottleneck_score"] = (
        0.35 * oee_penalty
        + 0.25 * avail_penalty
        + 0.20 * perf_penalty
        + 0.10 * qual_penalty
        + 0.10 * downtime_norm
    ).round(4)

    result["main_issue"] = result.apply(identify_main_issue, axis=1)
    return result.sort_values("bottleneck_score", ascending=False).reset_index(drop=True)
