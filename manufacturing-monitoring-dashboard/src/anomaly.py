"""Anomaly detection using Isolation Forest and rule-based alert generation."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import AnomalyConfig, AlertThresholds
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = AnomalyConfig()
_THRESHOLDS = AlertThresholds()


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = _CFG.default_contamination,
    min_rows: int = _CFG.min_rows_for_model,
) -> pd.DataFrame:
    """Run Isolation Forest on sensor features to flag anomalous rows.

    Args:
        df: Sensor DataFrame for a single machine (or multiple machines).
        contamination: Expected proportion of anomalies (0.01–0.15).
        min_rows: Minimum rows required to fit the model.

    Returns:
        DataFrame with added columns: anomaly_score, is_anomaly, anomaly_severity.
    """
    result = df.copy()
    result["anomaly_score"] = 0.0
    result["is_anomaly"] = False
    result["anomaly_severity"] = "normal"

    features = _CFG.features
    available = [f for f in features if f in result.columns]

    if len(result) < min_rows:
        logger.warning(
            "Too few rows (%d) for anomaly detection (min %d); returning defaults.",
            len(result),
            min_rows,
        )
        return result

    if not available:
        logger.warning("No feature columns available for anomaly detection.")
        return result

    feature_matrix = result[available].fillna(0.0)

    try:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        raw_scores = model.fit_predict(feature_matrix)
        decision_scores = model.decision_function(feature_matrix)

        result["is_anomaly"] = raw_scores == -1
        # Invert so higher = more anomalous, normalize to 0-1
        result["anomaly_score"] = (
            (-decision_scores - decision_scores.min())
            / (decision_scores.max() - decision_scores.min() + 1e-9)
        ).clip(0.0, 1.0)

    except Exception as exc:
        logger.error("Isolation Forest failed: %s", exc)
        return result

    result["anomaly_severity"] = result.apply(calculate_anomaly_severity, axis=1)
    n_anomalies = int(result["is_anomaly"].sum())
    logger.info("Anomaly detection complete: %d anomalies flagged.", n_anomalies)
    return result


def calculate_anomaly_severity(row: pd.Series) -> str:
    """Assign severity label to a row based on sensor thresholds.

    Args:
        row: A single sensor row with temperature_c, vibration_hz, anomaly_score.

    Returns:
        "high", "medium", "low", or "normal".
    """
    if not row.get("is_anomaly", False):
        return "normal"

    temp = row.get("temperature_c", 0.0)
    vib = row.get("vibration_hz", 0.0)
    score = row.get("anomaly_score", 0.0)

    if temp >= _THRESHOLDS.temperature_critical or vib >= _THRESHOLDS.vibration_critical or score > 0.8:
        return "high"
    if temp >= _THRESHOLDS.temperature_warning or vib >= _THRESHOLDS.vibration_warning or score > 0.5:
        return "medium"
    return "low"


def generate_anomaly_explanation(
    df: pd.DataFrame,
    anomaly_row: pd.Series,
) -> str:
    """Generate a human-readable explanation for why a point was flagged as anomalous.

    Uses z-score deviation from machine baseline as a proxy for Isolation Forest
    feature importance.

    Args:
        df: Historical baseline data for the machine (same machine_id).
        anomaly_row: The anomalous row to explain.

    Returns:
        Explanation string.
    """
    machine_id = anomaly_row.get("machine_id", "unknown")
    machine_df = df[df["machine_id"] == machine_id] if "machine_id" in df.columns else df

    signals: List[str] = []
    z_scores: dict = {}

    for col in ["temperature_c", "vibration_hz", "cycle_time_sec"]:
        if col not in machine_df.columns or col not in anomaly_row.index:
            continue
        baseline = machine_df[col].dropna()
        if len(baseline) < 5:
            continue
        mean = baseline.mean()
        std = baseline.std()
        if std < 1e-9:
            continue
        z = (anomaly_row[col] - mean) / std
        z_scores[col] = round(float(z), 2)
        if abs(z) > 2.0:
            direction = "above" if z > 0 else "below"
            signals.append(
                f"{col.replace('_', ' ').title()} is {abs(z):.1f}σ {direction} baseline "
                f"(current: {anomaly_row[col]:.1f}, baseline: {mean:.1f})"
            )

    if not signals:
        return (
            f"Machine {machine_id}: anomaly flagged by Isolation Forest based on the "
            "combined sensor profile. Individual deviations are within 2σ but the "
            "multivariate pattern is unusual."
        )

    main_signal = signals[0]
    explanation = (
        f"Machine {machine_id}: {main_signal}. "
    )
    if len(signals) > 1:
        explanation += "Additionally, " + "; ".join(signals[1:]) + ". "

    temp_z = z_scores.get("temperature_c")
    vib_z = z_scores.get("vibration_hz")
    ct_z = z_scores.get("cycle_time_sec")

    hints = []
    if vib_z and abs(vib_z) > 2.0 and ct_z and abs(ct_z) > 1.5:
        hints.append("elevated vibration with slower cycle time may indicate mechanical wear")
    if temp_z and abs(temp_z) > 2.0 and (vib_z is None or abs(vib_z) <= 1.5):
        hints.append("isolated temperature spike may indicate a cooling or load issue")
    if hints:
        explanation += "Possible cause: " + "; ".join(hints) + "."

    return explanation


def create_alert_feed(
    df: pd.DataFrame,
    selected_time: pd.Timestamp,
    max_alerts: int = 10,
) -> pd.DataFrame:
    """Generate a prioritized alert feed from sensor data up to selected_time.

    Args:
        df: Full sensor DataFrame.
        selected_time: Upper time bound for alerts.
        max_alerts: Maximum number of alerts to return.

    Returns:
        DataFrame with columns: timestamp, machine_id, severity, message.
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "machine_id", "severity", "message"])

    window = df[df["timestamp"] <= selected_time].copy()
    if window.empty:
        return pd.DataFrame(columns=["timestamp", "machine_id", "severity", "message"])

    alerts: List[dict] = []

    for _, row in window.iterrows():
        sev = _classify_alert_severity(row)
        if sev is None:
            continue
        msg = _format_alert_message(row, sev)
        alerts.append(
            {
                "timestamp": row["timestamp"],
                "machine_id": row["machine_id"],
                "severity": sev,
                "message": msg,
            }
        )

    if not alerts:
        return pd.DataFrame(columns=["timestamp", "machine_id", "severity", "message"])

    alert_df = (
        pd.DataFrame(alerts)
        .sort_values("timestamp", ascending=False)
        .head(max_alerts)
        .reset_index(drop=True)
    )
    return alert_df


def _classify_alert_severity(row: pd.Series) -> str | None:
    """Return alert severity for a row, or None if no alert condition met."""
    status = row.get("status", "running")
    temp = row.get("temperature_c", 0.0)
    vib = row.get("vibration_hz", 0.0)
    util = row.get("utilization_pct", 100.0)

    if status == "down":
        return "High"
    if temp >= _THRESHOLDS.temperature_critical:
        return "High"
    if vib >= _THRESHOLDS.vibration_critical:
        return "High"
    if _THRESHOLDS.temperature_warning <= temp < _THRESHOLDS.temperature_critical:
        return "Medium"
    if _THRESHOLDS.vibration_warning <= vib < _THRESHOLDS.vibration_critical:
        return "Medium"
    if status == "idle" and util < _THRESHOLDS.utilization_low:
        return "Low"
    return None


def _format_alert_message(row: pd.Series, severity: str) -> str:
    """Build a human-readable alert message string."""
    machine_id = row.get("machine_id", "unknown")
    status = row.get("status", "running")
    temp = row.get("temperature_c", 0.0)
    vib = row.get("vibration_hz", 0.0)
    util = row.get("utilization_pct", 0.0)

    if status == "down":
        reason = row.get("downtime_reason") or "unknown"
        return f"{machine_id} is DOWN — reason: {reason}"
    if temp >= _THRESHOLDS.temperature_critical:
        return f"{machine_id} temperature reached {temp:.1f}°C (critical)"
    if vib >= _THRESHOLDS.vibration_critical:
        return f"{machine_id} vibration exceeded {vib:.1f} Hz (critical)"
    if temp >= _THRESHOLDS.temperature_warning:
        return f"{machine_id} temperature reached {temp:.1f}°C (warning)"
    if vib >= _THRESHOLDS.vibration_warning:
        return f"{machine_id} vibration reached {vib:.1f} Hz (warning)"
    return f"{machine_id} utilization dropped to {util:.1f}%"
