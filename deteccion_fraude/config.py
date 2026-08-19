import os
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


class ExperimentConfig:
    """Configuracion central del experimento de deteccion de fraude."""

    sequence_length: int = 5
    smote_ratio: float = 0.3
    random_state: int = 42
    false_negative_cost: float = 150.0
    false_positive_cost: float = 25.0
    min_lstm_features: int = 30
    models_dir: Path = MODELS_DIR

    data_raw: Path = RAW_DATA_DIR / "dataset.csv"

    robust_columns: ClassVar[list[str]] = [
        "Amount_log",
        "Amount_log_cat",
        "Amount_zscore",
        "Amount_roll_mean_5",
        "Amount_roll_std_5",
        "Amount_roll_mean_10",
        "Amount_roll_ratio",
        "Amount_V_ratio",
        "Amount_V_std_ratio",
        "V_mean",
        "V_std",
        "V_max",
        "V_min",
        "V_median",
        "V_abs_sum",
    ]

    standard_columns: ClassVar[list[str]] = [
        "Time_hour",
        "Transaction_frequency",
        "V1_V2_ratio",
        "V3_V4_ratio",
        "V12_V14_ratio",
        "V_count_neg",
        "V_count_pos",
    ]

    feature_columns: ClassVar[list[str]] = (
        [f"V{i}" for i in range(1, 29)] + robust_columns + standard_columns
    )

    def create_output_directories(self) -> None:
        """Crea directorios de salida para modelos y reportes."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
