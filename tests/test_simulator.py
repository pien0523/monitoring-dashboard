"""Tests for the synthetic data simulator."""

import pandas as pd
import pytest

from src.config import SimulationConfig
from src.schemas import REQUIRED_COLUMNS
from src.simulator import generate_synthetic_data


@pytest.fixture(scope="module")
def sim_config() -> SimulationConfig:
    return SimulationConfig(n_machines=7, n_days=2, random_seed=42)


@pytest.fixture(scope="module")
def sim_df(sim_config: SimulationConfig) -> pd.DataFrame:
    return generate_synthetic_data(sim_config)


def test_simulator_returns_nonempty_dataframe(sim_df: pd.DataFrame) -> None:
    """Simulator must return a non-empty DataFrame."""
    assert isinstance(sim_df, pd.DataFrame)
    assert len(sim_df) > 0


def test_simulator_has_required_columns(sim_df: pd.DataFrame) -> None:
    """Simulator output must contain every column in REQUIRED_COLUMNS."""
    for col in REQUIRED_COLUMNS:
        assert col in sim_df.columns, f"Missing required column: {col}"


def test_simulator_generates_correct_machines(sim_df: pd.DataFrame) -> None:
    """Simulator must produce exactly 7 distinct machine IDs."""
    machine_ids = sim_df["machine_id"].unique()
    assert len(machine_ids) == 7
    for i in range(1, 8):
        assert f"M-{str(i).zfill(3)}" in machine_ids


def test_simulator_row_count(sim_df: pd.DataFrame, sim_config: SimulationConfig) -> None:
    """Row count must equal n_machines × n_days × 24 × 60."""
    expected = sim_config.n_machines * sim_config.n_days * 24 * 60
    assert len(sim_df) == expected


def test_simulator_creates_needs_maintenance_column(sim_df: pd.DataFrame) -> None:
    """needs_maintenance column must exist and contain only 0 or 1."""
    assert "needs_maintenance" in sim_df.columns
    unique_values = set(sim_df["needs_maintenance"].unique())
    assert unique_values.issubset({0, 1})


def test_simulator_timestamp_is_monotonic(sim_df: pd.DataFrame) -> None:
    """Timestamps within each machine must be monotonically increasing."""
    for machine_id, grp in sim_df.groupby("machine_id"):
        assert grp["timestamp"].is_monotonic_increasing, (
            f"Timestamps for {machine_id} are not monotonically increasing"
        )


def test_simulator_status_values_are_valid(sim_df: pd.DataFrame) -> None:
    """Status column must only contain allowed values."""
    valid = {"running", "idle", "down", "maintenance"}
    assert set(sim_df["status"].unique()).issubset(valid)


def test_simulator_shift_values_are_valid(sim_df: pd.DataFrame) -> None:
    """Shift column must only contain 'day' or 'night'."""
    assert set(sim_df["shift"].unique()).issubset({"day", "night"})
