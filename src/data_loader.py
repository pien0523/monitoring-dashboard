"""Data loading and validation utilities."""

from __future__ import annotations

import os

import pandas as pd

from src.config import SimulationConfig
from src.logger import get_logger
from src.schemas import REQUIRED_COLUMNS

logger = get_logger(__name__)

_DEFAULT_PATH = "data/machine_sensor_data.csv"


def load_machine_data(path: str = _DEFAULT_PATH) -> pd.DataFrame:
    """Load sensor data from CSV and parse timestamps.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with parsed timestamps, sorted by machine and time.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Assert that all required schema columns are present.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_or_create_data(path: str = _DEFAULT_PATH) -> pd.DataFrame:
    """Return the sensor dataset, generating it if the CSV does not exist.

    Args:
        path: Path to look for (or save) the CSV file.

    Returns:
        Validated sensor DataFrame.
    """
    if not os.path.exists(path):
        logger.warning("Data file not found at %s — generating synthetic data.", path)
        from src.simulator import generate_synthetic_data  # avoid circular at module load

        config = SimulationConfig(output_path=path)
        df = generate_synthetic_data(config)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved generated data to %s", path)
    else:
        df = load_machine_data(path)

    validate_required_columns(df)
    return df
