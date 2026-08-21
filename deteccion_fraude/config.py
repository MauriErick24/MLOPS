"""Configuracion y rutas del experimento.

Sigue la ubicacion de Cookiecutter Data Science v2.
"""

from dataclasses import dataclass
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

try:
    from loguru import logger
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    """Parametros inmutables y reproducibles del experimento."""

    data_file: Path = RAW_DATA_DIR / "creditcard.csv"
    models_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR
    random_state: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    sequence_length: int = 5
    smote_ratio: float = 0.30
    false_negative_cost: float = 150.0
    false_positive_cost: float = 25.0
    min_lstm_features: int = 30
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "deteccion_fraude"

    @property
    def pca_columns(self) -> list[str]:
        return [f"V{i}" for i in range(1, 29)]

    @property
    def robust_columns(self) -> list[str]:
        return [
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

    @property
    def standard_columns(self) -> list[str]:
        return [
            "Time_hour",
            "Transaction_frequency",
            "V1_V2_ratio",
            "V3_V4_ratio",
            "V12_V14_ratio",
            "V_count_neg",
            "V_count_pos",
        ]

    @property
    def feature_columns(self) -> list[str]:
        return self.pca_columns + self.robust_columns + self.standard_columns

    def create_output_directories(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
