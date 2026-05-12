"""Central configuration for the manufacturing monitoring dashboard."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimulationConfig:
    n_machines: int = 7
    n_days: int = 7
    frequency_minutes: int = 1
    random_seed: int = 42
    start_date: str = "2024-01-01"
    output_path: str = "data/machine_sensor_data.csv"


@dataclass
class OEEConfig:
    ideal_cycle_time_sec: float = 24.0
    max_performance_cap: float = 1.2
    oee_good_threshold: float = 0.85
    oee_watch_threshold: float = 0.70


@dataclass
class AnomalyConfig:
    default_contamination: float = 0.05
    min_contamination: float = 0.01
    max_contamination: float = 0.15
    min_rows_for_model: int = 30
    features: List[str] = field(default_factory=lambda: [
        "temperature_c", "vibration_hz", "utilization_pct", "cycle_time_sec"
    ])


@dataclass
class MaintenanceConfig:
    planned_maintenance_cost: float = 1500.0
    unplanned_failure_cost: float = 8000.0
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.40
    health_score_critical: float = 60.0
    health_score_warning: float = 80.0
    rolling_window_hours: int = 6


@dataclass
class AlertThresholds:
    temperature_critical: float = 85.0
    temperature_warning: float = 78.0
    vibration_critical: float = 18.0
    vibration_warning: float = 15.0
    oee_critical: float = 0.70
    failure_prob_critical: float = 0.75
    anomaly_count_warning: int = 5
    downtime_minutes_warning: int = 60
    utilization_low: float = 50.0


# Machine profiles define baseline behavior for each machine
MACHINE_PROFILES: Dict[str, Dict] = {
    "M-001": {
        "story": "healthy_stable",
        "base_temp": 65.0,
        "base_vibration": 10.5,
        "base_cycle_time": 24.0,
        "initial_degradation": 0.10,
        "degradation_rate": 0.003,
        "downtime_probability": 0.002,
        "age_factor": 0.2,
    },
    "M-002": {
        "story": "high_load",
        "base_temp": 70.0,
        "base_vibration": 11.5,
        "base_cycle_time": 25.0,
        "initial_degradation": 0.25,
        "degradation_rate": 0.008,
        "downtime_probability": 0.005,
        "age_factor": 0.5,
    },
    "M-003": {
        "story": "degrading_predictive",
        "base_temp": 68.0,
        "base_vibration": 12.0,
        "base_cycle_time": 26.0,
        "initial_degradation": 0.35,
        "degradation_rate": 0.060,
        "downtime_probability": 0.010,
        "age_factor": 0.8,
    },
    "M-004": {
        "story": "bottleneck",
        "base_temp": 66.0,
        "base_vibration": 11.0,
        "base_cycle_time": 38.0,
        "initial_degradation": 0.20,
        "degradation_rate": 0.005,
        "downtime_probability": 0.004,
        "age_factor": 0.6,
    },
    "M-005": {
        "story": "cooling_issue",
        "base_temp": 72.0,
        "base_vibration": 10.8,
        "base_cycle_time": 25.0,
        "initial_degradation": 0.22,
        "degradation_rate": 0.007,
        "downtime_probability": 0.004,
        "age_factor": 0.4,
    },
    "M-006": {
        "story": "control_normal",
        "base_temp": 64.0,
        "base_vibration": 10.2,
        "base_cycle_time": 24.5,
        "initial_degradation": 0.12,
        "degradation_rate": 0.003,
        "downtime_probability": 0.002,
        "age_factor": 0.3,
    },
    "M-007": {
        "story": "unstable_downtime",
        "base_temp": 67.0,
        "base_vibration": 12.5,
        "base_cycle_time": 27.0,
        "initial_degradation": 0.30,
        "degradation_rate": 0.010,
        "downtime_probability": 0.035,
        "age_factor": 0.7,
    },
}

# Downtime block definitions: (machine_id, day_offset, start_hour, start_min, duration_minutes, reason)
DOWNTIME_BLOCKS = [
    ("M-007", 0, 10, 20, 35, "machine_failure"),
    ("M-007", 1, 14, 45, 60, "machine_failure"),
    ("M-007", 2, 8,  10, 25, "operator_unavailable"),
    ("M-007", 3, 16, 30, 50, "machine_failure"),
    ("M-007", 4, 11, 0,  40, "material_shortage"),
    ("M-007", 5, 9,  15, 55, "machine_failure"),
    ("M-007", 6, 13, 0,  45, "machine_failure"),
    ("M-003", 2, 2,  0,  90, "planned_maintenance"),
    ("M-003", 5, 3,  0,  60, "planned_maintenance"),
    ("M-005", 1, 13, 30, 20, "quality_hold"),
    ("M-005", 3, 10, 0,  15, "quality_hold"),
    ("M-004", 0, 9,  0,  20, "material_shortage"),
    ("M-004", 3, 15, 0,  30, "material_shortage"),
    ("M-002", 2, 7,  30, 15, "machine_failure"),
    ("M-001", 4, 11, 0,  10, "planned_maintenance"),
]

# Anomaly spike definitions: (machine_id, day_offset, start_hour, duration_minutes, type)
ANOMALY_SPIKES = [
    ("M-005", 0, 13, 15, "temp_spike"),
    ("M-005", 1, 14, 20, "temp_spike"),
    ("M-005", 2, 11, 12, "temp_spike"),
    ("M-005", 4, 15, 18, "temp_spike"),
    ("M-005", 6, 13, 10, "temp_spike"),
    ("M-003", 1, 16, 10, "vibration_spike"),
    ("M-003", 3, 9,  15, "vibration_spike"),
    ("M-003", 5, 12, 20, "vibration_spike"),
    ("M-002", 2, 10, 12, "temp_spike"),
    ("M-007", 1, 17, 10, "vibration_spike"),
]

# Time constants
DAY_SHIFT_START: int = 7
DAY_SHIFT_END: int = 19
SLIDER_STEP_MINUTES: int = 15

# Rolling window definitions (hours)
ROLLING_WINDOWS = {
    "recent": 6,
    "sensor": 24,
    "trend": 7 * 24,
    "anomaly_lookback": 6,
}
