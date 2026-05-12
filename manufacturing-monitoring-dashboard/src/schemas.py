"""Data schemas and dataclasses for the manufacturing monitoring dashboard."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MachineReading:
    """Represents a single sensor reading row from a machine."""

    timestamp: str
    machine_id: str
    status: str
    shift: str
    utilization_pct: float
    temperature_c: float
    vibration_hz: float
    cycle_time_sec: float
    output_count: int
    defect_count: int
    machine_age_factor: float
    latent_degradation: float
    event_type: str
    downtime_reason: Optional[str]
    needs_maintenance: int


@dataclass
class MachineSummary:
    """Aggregated health summary for a single machine."""

    machine_id: str
    health_score: float
    failure_probability_7d: float
    avg_temperature_6h: float
    avg_vibration_6h: float
    avg_cycle_time_6h: float
    defect_rate_6h: float
    downtime_minutes_6h: float
    anomaly_count_6h: int
    machine_age_factor: float
    latent_degradation: float
    oee: float
    recommendation: str = "Continue normal monitoring"
    confidence: float = 0.0
    expected_savings: float = 0.0


@dataclass
class MaintenanceRecommendation:
    """Maintenance action recommendation for a machine."""

    machine_id: str
    health_score: float
    failure_probability_7d: float
    recommendation: str
    confidence: float
    expected_savings: float
    priority: int = 0


REQUIRED_COLUMNS = [
    "timestamp",
    "machine_id",
    "status",
    "shift",
    "utilization_pct",
    "temperature_c",
    "vibration_hz",
    "cycle_time_sec",
    "output_count",
    "defect_count",
    "machine_age_factor",
    "latent_degradation",
    "event_type",
    "downtime_reason",
    "needs_maintenance",
]

VALID_STATUSES = {"running", "idle", "down", "maintenance"}
VALID_SHIFTS = {"day", "night"}
VALID_EVENT_TYPES = {"normal", "temp_spike", "vibration_spike", "downtime", "maintenance"}
VALID_DOWNTIME_REASONS = {
    "machine_failure",
    "material_shortage",
    "quality_hold",
    "planned_maintenance",
    "operator_unavailable",
    None,
}
