"""Predictive maintenance model, health scoring, and feature engineering."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import MaintenanceConfig
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = MaintenanceConfig()

FEATURE_COLS = [
    "avg_temperature_6h",
    "max_temperature_6h",
    "avg_vibration_6h",
    "max_vibration_6h",
    "avg_cycle_time_6h",
    "defect_rate_6h",
    "downtime_minutes_6h",
    "anomaly_count_6h",
    "machine_age_factor",
    "latent_degradation",
]


def create_rolling_features(
    df: pd.DataFrame,
    selected_time: pd.Timestamp,
) -> pd.DataFrame:
    """Build per-machine rolling feature vectors for the 6-hour window before selected_time.

    Args:
        df: Full sensor DataFrame sorted by machine_id and timestamp.
        selected_time: Upper time bound for the rolling window.

    Returns:
        DataFrame with one row per machine and FEATURE_COLS + needs_maintenance.
    """
    if df.empty:
        logger.warning("create_rolling_features received empty dataframe.")
        return pd.DataFrame(columns=FEATURE_COLS + ["machine_id", "needs_maintenance"])

    window_start = selected_time - pd.Timedelta(hours=_CFG.rolling_window_hours)
    recent = df[(df["timestamp"] > window_start) & (df["timestamp"] <= selected_time)]

    if recent.empty:
        logger.warning("No rows in the 6-hour rolling window for selected_time %s.", selected_time)
        recent = df[df["timestamp"] <= selected_time]

    records = []
    for machine_id, grp in recent.groupby("machine_id"):
        running = grp[grp["status"] == "running"]

        avg_temp = grp["temperature_c"].mean()
        max_temp = grp["temperature_c"].max()
        avg_vib = grp["vibration_hz"].mean()
        max_vib = grp["vibration_hz"].max()

        avg_ct = running["cycle_time_sec"].mean() if not running.empty else 0.0
        total_output = grp["output_count"].sum()
        total_defect = grp["defect_count"].sum()
        defect_rate = float(total_defect / total_output) if total_output > 0 else 0.0

        downtime_minutes = float(len(grp[grp["status"] == "down"]))
        anomaly_count = int(grp.get("is_anomaly", pd.Series(False, index=grp.index)).sum())

        age_factor = grp["machine_age_factor"].iloc[-1]
        degradation = grp["latent_degradation"].iloc[-1]
        needs_maint = int(grp["needs_maintenance"].iloc[-1])

        records.append(
            {
                "machine_id": machine_id,
                "avg_temperature_6h": round(float(avg_temp), 2),
                "max_temperature_6h": round(float(max_temp), 2),
                "avg_vibration_6h": round(float(avg_vib), 3),
                "max_vibration_6h": round(float(max_vib), 3),
                "avg_cycle_time_6h": round(float(avg_ct) if not np.isnan(avg_ct) else 0.0, 2),
                "defect_rate_6h": round(defect_rate, 4),
                "downtime_minutes_6h": downtime_minutes,
                "anomaly_count_6h": anomaly_count,
                "machine_age_factor": float(age_factor),
                "latent_degradation": round(float(degradation), 4),
                "needs_maintenance": needs_maint,
            }
        )

    return pd.DataFrame(records).reset_index(drop=True)


def train_maintenance_model(df: pd.DataFrame) -> Optional[Any]:
    """Train a Random Forest classifier to predict maintenance need.

    Falls back gracefully when data is too sparse or has only one class.

    Args:
        df: Full sensor DataFrame (all time) used as training set.

    Returns:
        Fitted RandomForestClassifier, or None if training is not possible.
    """
    if df.empty:
        logger.warning("Cannot train model on empty dataframe.")
        return None

    # Use all historical data to build features for training
    all_times = df["timestamp"].sort_values().unique()
    # Sample every 60 minutes to avoid O(n) growth
    step = 60
    sample_times = all_times[::step]

    rows = []
    for t in sample_times:
        ts = pd.Timestamp(t)
        window_start = ts - pd.Timedelta(hours=_CFG.rolling_window_hours)
        window = df[(df["timestamp"] > window_start) & (df["timestamp"] <= ts)]
        if window.empty:
            continue
        for machine_id, grp in window.groupby("machine_id"):
            running = grp[grp["status"] == "running"]
            avg_temp = grp["temperature_c"].mean()
            max_temp = grp["temperature_c"].max()
            avg_vib = grp["vibration_hz"].mean()
            max_vib = grp["vibration_hz"].max()
            avg_ct = running["cycle_time_sec"].mean() if not running.empty else 0.0
            total_output = grp["output_count"].sum()
            total_defect = grp["defect_count"].sum()
            defect_rate = float(total_defect / total_output) if total_output > 0 else 0.0
            downtime_min = float(len(grp[grp["status"] == "down"]))
            anomaly_ct = int(grp.get("is_anomaly", pd.Series(False)).sum())
            age_factor = grp["machine_age_factor"].iloc[-1]
            degradation = grp["latent_degradation"].iloc[-1]
            label = int(grp["needs_maintenance"].iloc[-1])

            rows.append({
                "avg_temperature_6h": float(avg_temp),
                "max_temperature_6h": float(max_temp),
                "avg_vibration_6h": float(avg_vib),
                "max_vibration_6h": float(max_vib),
                "avg_cycle_time_6h": float(avg_ct) if not np.isnan(avg_ct) else 0.0,
                "defect_rate_6h": defect_rate,
                "downtime_minutes_6h": downtime_min,
                "anomaly_count_6h": float(anomaly_ct),
                "machine_age_factor": float(age_factor),
                "latent_degradation": float(degradation),
                "needs_maintenance": label,
            })

    if not rows:
        logger.warning("No training samples generated.")
        return None

    train_df = pd.DataFrame(rows).dropna()
    X = train_df[FEATURE_COLS].values
    y = train_df["needs_maintenance"].values

    if len(np.unique(y)) < 2:
        logger.warning("Only one target class in training data; skipping model training.")
        return None

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    try:
        model.fit(X, y)
        logger.info("Random Forest trained on %d samples.", len(y))
        return model
    except Exception as exc:
        logger.error("Model training failed: %s", exc)
        return None


def predict_failure_probability(
    features_df: pd.DataFrame,
    model: Optional[Any],
) -> pd.DataFrame:
    """Predict 7-day failure probability for each machine.

    Falls back to rule-based probability when model is None.

    Args:
        features_df: DataFrame from create_rolling_features.
        model: Fitted RandomForestClassifier or None.

    Returns:
        features_df with added failure_probability_7d column.
    """
    result = features_df.copy()

    if result.empty:
        result["failure_probability_7d"] = []
        return result

    available = [c for c in FEATURE_COLS if c in result.columns]
    X = result[available].fillna(0.0).values

    if model is not None:
        try:
            probs = model.predict_proba(X)
            # Column index for class=1
            classes = list(model.classes_)
            pos_idx = classes.index(1) if 1 in classes else -1
            if pos_idx >= 0:
                result["failure_probability_7d"] = probs[:, pos_idx]
            else:
                result["failure_probability_7d"] = _rule_based_probability(result)
        except Exception as exc:
            logger.error("Prediction failed: %s; using rule-based fallback.", exc)
            result["failure_probability_7d"] = _rule_based_probability(result)
    else:
        logger.info("No model available; using rule-based probability fallback.")
        result["failure_probability_7d"] = _rule_based_probability(result)

    result["failure_probability_7d"] = result["failure_probability_7d"].clip(0.0, 1.0)
    return result


def _rule_based_probability(features_df: pd.DataFrame) -> pd.Series:
    """Compute a normalized risk score without a trained model."""
    df = features_df.copy()

    temp_score = ((df.get("avg_temperature_6h", 0) - 60) / 30).clip(0, 1)
    vib_score = ((df.get("avg_vibration_6h", 0) - 10) / 10).clip(0, 1)
    defect_score = (df.get("defect_rate_6h", 0) / 0.15).clip(0, 1)
    downtime_score = (df.get("downtime_minutes_6h", 0) / 60).clip(0, 1)
    anomaly_score = (df.get("anomaly_count_6h", 0) / 10).clip(0, 1)
    degradation_score = df.get("latent_degradation", 0).clip(0, 1)

    combined = (
        0.20 * temp_score
        + 0.25 * vib_score
        + 0.15 * defect_score
        + 0.15 * downtime_score
        + 0.10 * anomaly_score
        + 0.15 * degradation_score
    )
    return combined.clip(0.0, 1.0)


def calculate_health_score(row: pd.Series) -> float:
    """Compute a 0–100 health score from risk components.

    Args:
        row: A features row with failure_probability_7d and sensor metrics.

    Returns:
        Health score clamped to [0, 100].
    """
    failure_prob = row.get("failure_probability_7d", 0.0)
    vib = row.get("avg_vibration_6h", 10.0)
    temp = row.get("avg_temperature_6h", 65.0)
    defect_rate = row.get("defect_rate_6h", 0.0)
    downtime = row.get("downtime_minutes_6h", 0.0)
    anomaly_count = row.get("anomaly_count_6h", 0)

    vib_risk = float(np.clip((vib - 10) / 10, 0, 1))
    temp_risk = float(np.clip((temp - 60) / 30, 0, 1))
    defect_risk = float(np.clip(defect_rate / 0.15, 0, 1))
    downtime_risk = float(np.clip(downtime / 60, 0, 1))
    anomaly_risk = float(np.clip(anomaly_count / 10, 0, 1))

    weighted_risk = (
        0.30 * failure_prob
        + 0.20 * vib_risk
        + 0.20 * temp_risk
        + 0.10 * defect_risk
        + 0.10 * downtime_risk
        + 0.10 * anomaly_risk
    )
    return float(np.clip(100.0 - weighted_risk * 100.0, 0.0, 100.0))


def generate_machine_health_summary(
    df: pd.DataFrame,
    selected_time: pd.Timestamp,
    model: Optional[Any] = None,
) -> pd.DataFrame:
    """Produce a health summary DataFrame for all machines at selected_time.

    Args:
        df: Full sensor DataFrame.
        selected_time: Time reference for rolling windows.
        model: Optional trained Random Forest model.

    Returns:
        DataFrame with health scores, failure probabilities, and features per machine.
    """
    if df.empty:
        logger.warning("generate_machine_health_summary received empty dataframe.")
        return pd.DataFrame()

    features_df = create_rolling_features(df, selected_time)

    if features_df.empty:
        return features_df

    features_df = predict_failure_probability(features_df, model)
    features_df["health_score"] = features_df.apply(calculate_health_score, axis=1).round(1)

    return features_df.reset_index(drop=True)
