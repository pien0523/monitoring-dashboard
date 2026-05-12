"""Synthetic manufacturing sensor data generator.

Run directly to generate and save the dataset:
    python -m src.simulator
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ANOMALY_SPIKES,
    DAY_SHIFT_END,
    DAY_SHIFT_START,
    DOWNTIME_BLOCKS,
    MACHINE_PROFILES,
    SimulationConfig,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_downtime_index(
    config: SimulationConfig,
    start_dt: datetime,
) -> Dict[Tuple[str, datetime], Tuple[str, str]]:
    """Pre-compute a mapping of (machine_id, minute_dt) -> (status, reason) for downtime blocks."""
    index: Dict[Tuple[str, datetime], Tuple[str, str]] = {}
    status_map = {
        "machine_failure": "down",
        "material_shortage": "idle",
        "quality_hold": "idle",
        "planned_maintenance": "maintenance",
        "operator_unavailable": "idle",
    }
    for machine_id, day_offset, start_h, start_m, duration, reason in DOWNTIME_BLOCKS:
        block_start = start_dt + timedelta(
            days=day_offset, hours=start_h, minutes=start_m
        )
        block_status = status_map.get(reason, "down")
        for minute in range(duration):
            dt = block_start + timedelta(minutes=minute)
            index[(machine_id, dt)] = (block_status, reason)
    return index


def _build_spike_index(
    config: SimulationConfig,
    start_dt: datetime,
) -> Dict[Tuple[str, datetime], str]:
    """Pre-compute a mapping of (machine_id, minute_dt) -> spike_type for anomaly spikes."""
    index: Dict[Tuple[str, datetime], str] = {}
    for machine_id, day_offset, start_h, duration, spike_type in ANOMALY_SPIKES:
        block_start = start_dt + timedelta(days=day_offset, hours=start_h)
        for minute in range(duration):
            dt = block_start + timedelta(minutes=minute)
            index[(machine_id, dt)] = spike_type
    return index


def _compute_degradation(
    profile: Dict,
    elapsed_minutes: int,
    maintenance_reductions: List[int],
) -> float:
    """Compute latent degradation at a given elapsed-minutes point."""
    total_minutes = elapsed_minutes
    reduction = sum(
        0.15 for m in maintenance_reductions if m <= elapsed_minutes
    )
    raw = profile["initial_degradation"] + profile["degradation_rate"] * (total_minutes / 60.0)
    return float(np.clip(raw - reduction, 0.0, 1.0))


def _get_shift(hour: int) -> str:
    return "day" if DAY_SHIFT_START <= hour < DAY_SHIFT_END else "night"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(config: SimulationConfig) -> pd.DataFrame:
    """Generate a synthetic sensor dataset for all machines over the configured period.

    Args:
        config: SimulationConfig specifying n_machines, n_days, frequency, seed, etc.

    Returns:
        DataFrame with one row per machine per minute, schema matching REQUIRED_COLUMNS.
    """
    rng = np.random.default_rng(config.random_seed)
    start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")

    machine_ids = [f"M-{str(i).zfill(3)}" for i in range(1, config.n_machines + 1)]
    total_minutes = config.n_days * 24 * 60

    downtime_idx = _build_downtime_index(config, start_dt)
    spike_idx = _build_spike_index(config, start_dt)

    # Track when planned_maintenance events happen per machine for degradation reset
    maintenance_events: Dict[str, List[int]] = {m: [] for m in machine_ids}
    for machine_id, day_offset, start_h, start_m, _, reason in DOWNTIME_BLOCKS:
        if reason == "planned_maintenance":
            elapsed = day_offset * 24 * 60 + start_h * 60 + start_m
            maintenance_events[machine_id].append(elapsed)

    all_rows: List[Dict] = []
    logger.info(
        "Generating synthetic data: %d machines × %d days (%d rows expected)",
        config.n_machines,
        config.n_days,
        config.n_machines * total_minutes,
    )

    for machine_id in machine_ids:
        profile = MACHINE_PROFILES[machine_id]
        story = profile["story"]
        m_events = maintenance_events[machine_id]

        for minute_offset in range(total_minutes):
            dt = start_dt + timedelta(minutes=minute_offset)
            hour = dt.hour
            shift = _get_shift(hour)

            # Determine degradation
            degradation = _compute_degradation(profile, minute_offset, m_events)

            # Downtime block override
            downtime_key = (machine_id, dt)
            forced_status: str | None = None
            downtime_reason: str | None = None
            if downtime_key in downtime_idx:
                forced_status, downtime_reason = downtime_idx[downtime_key]

            # Determine status
            if forced_status:
                status = forced_status
            else:
                down_prob = profile["downtime_probability"] + 0.02 * degradation
                r = rng.random()
                if r < down_prob:
                    status = "down"
                    downtime_reason = rng.choice(
                        ["machine_failure", "operator_unavailable"]
                    )
                elif r < down_prob + 0.05:
                    status = "idle"
                else:
                    status = "running"

            # Utilization
            if status == "running":
                if shift == "day":
                    base_util = rng.normal(85, 5)
                else:
                    base_util = rng.normal(60, 8)
                if story == "high_load":
                    base_util += 5
                utilization = float(np.clip(base_util, 50, 100))
            elif status == "idle":
                utilization = float(rng.uniform(0, 20))
            else:
                utilization = 0.0

            # Spike type
            spike_type = spike_idx.get((machine_id, dt), None)

            # Temperature
            shift_effect = 2.0 if shift == "day" else -1.0
            temp = (
                profile["base_temp"]
                + 0.12 * utilization
                + 15.0 * degradation
                + shift_effect
                + rng.normal(0, 1.2)
            )
            if spike_type == "temp_spike":
                temp += rng.uniform(10, 17)
            if story == "cooling_issue" and utilization > 80:
                temp += rng.uniform(5, 12)
            temperature = float(np.clip(temp, 45, 98))

            # Vibration
            vib = (
                profile["base_vibration"]
                + 10.0 * degradation
                + 0.03 * utilization
                + rng.normal(0, 0.5)
            )
            if spike_type == "vibration_spike":
                vib += rng.uniform(6, 10)
            vibration = float(np.clip(vib, 5, 30))

            # Cycle time
            if status == "running":
                ct = (
                    profile["base_cycle_time"]
                    + 12.0 * degradation
                    + rng.normal(0, 0.8)
                )
                cycle_time = float(max(ct, 5.0))
            else:
                cycle_time = 0.0

            # Output count
            if status == "running" and cycle_time > 0:
                output_count = int((60.0 / cycle_time) * (utilization / 100.0))
            else:
                output_count = 0

            # Defect count
            if output_count > 0:
                defect_rate = 0.015 + 0.05 * degradation
                if temperature > 80:
                    defect_rate += 0.03
                if vibration > 16:
                    defect_rate += 0.04
                defect_rate = float(np.clip(defect_rate, 0.0, 0.5))
                defect_count = int(rng.binomial(output_count, defect_rate))
            else:
                defect_count = 0

            # Event type
            if forced_status == "maintenance":
                event_type = "maintenance"
            elif forced_status in ("down", "idle") and downtime_reason:
                event_type = "downtime"
            elif spike_type == "temp_spike":
                event_type = "temp_spike"
            elif spike_type == "vibration_spike":
                event_type = "vibration_spike"
            else:
                event_type = "normal"

            all_rows.append(
                {
                    "timestamp": dt,
                    "machine_id": machine_id,
                    "status": status,
                    "shift": shift,
                    "utilization_pct": round(utilization, 2),
                    "temperature_c": round(temperature, 2),
                    "vibration_hz": round(vibration, 3),
                    "cycle_time_sec": round(cycle_time, 2),
                    "output_count": output_count,
                    "defect_count": defect_count,
                    "machine_age_factor": profile["age_factor"],
                    "latent_degradation": round(degradation, 4),
                    "event_type": event_type,
                    "downtime_reason": downtime_reason,
                    "needs_maintenance": 0,  # computed below
                }
            )

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _compute_needs_maintenance(df)

    logger.info("Dataset generated: %d rows", len(df))
    return df


def _compute_needs_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based needs_maintenance column.

    Rules:
    - latent_degradation > 0.70
    - rolling-6h avg vibration > 15
    - rolling-6h avg temperature > 78
    - anomaly count in past 6h >= 5  (approximated by vibration/temp spikes)
    - downtime minutes in past 24h >= 60
    """
    df = df.copy()
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    window_6h = 6 * 60
    window_24h = 24 * 60

    results = []
    for machine_id, grp in df.groupby("machine_id", sort=False):
        grp = grp.reset_index(drop=True)

        roll_vib = grp["vibration_hz"].rolling(window_6h, min_periods=1).mean()
        roll_temp = grp["temperature_c"].rolling(window_6h, min_periods=1).mean()

        is_spike = (grp["event_type"].isin(["temp_spike", "vibration_spike"])).astype(int)
        anomaly_count_6h = is_spike.rolling(window_6h, min_periods=1).sum()

        is_down = (grp["status"] == "down").astype(int)
        downtime_24h = is_down.rolling(window_24h, min_periods=1).sum()

        needs = (
            (grp["latent_degradation"] > 0.70)
            | (roll_vib > 15.0)
            | (roll_temp > 78.0)
            | (anomaly_count_6h >= 5)
            | (downtime_24h >= 60)
        ).astype(int)

        grp["needs_maintenance"] = needs
        results.append(grp)

    return pd.concat(results).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate synthetic data and save to CSV."""
    config = SimulationConfig()
    df = generate_synthetic_data(config)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    df.to_csv(config.output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), config.output_path)


if __name__ == "__main__":
    main()
