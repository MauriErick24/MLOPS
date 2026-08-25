"""Artefactos de inferencia: puente entre el entrenamiento y la API.

El Model Registry de MLflow guarda el modelo, pero no el preprocesamiento que
lo alimenta. Sin los scalers ajustados, la mascara de ruido y los umbrales de
decision, un modelo en Production no puede puntuar una transaccion cruda. Este
modulo persiste ese estado y lo reaplica en inferencia.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Literal

from loguru import logger
import numpy as np
import pandas as pd

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.features import AmountStats, FraudFeatureEngineer

ARTIFACTS_FILENAME = "serving_artifacts.pkl"

RAW_COLUMNS: list[str] = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]

ModelKind = Literal["tabnet", "lstm", "pyfunc"]


@dataclass
class ServingArtifacts:
    """Estado de preprocesamiento necesario para puntuar transacciones crudas.

    Las listas de columnas se persisten junto a los scalers para que el
    artefacto sea autodescriptivo: un cambio posterior en `ExperimentConfig`
    no altera silenciosamente el contrato de un modelo ya promovido.
    """

    robust_scaler: Any
    standard_scaler: Any
    amount_stats: AmountStats
    robust_columns: list[str]
    standard_columns: list[str]
    feature_columns: list[str]
    noise_mask: np.ndarray
    feature_names: list[str]
    selected_features: list[str]
    thresholds: dict[str, float]
    sequence_length: int

    @classmethod
    def from_training(
        cls, config: ExperimentConfig, data, selector, results: dict
    ) -> "ServingArtifacts":
        """Construye el artefacto al final del entrenamiento."""
        return cls(
            robust_scaler=data.robust_scaler,
            standard_scaler=data.standard_scaler,
            amount_stats=data.amount_stats,
            robust_columns=list(config.robust_columns),
            standard_columns=list(config.standard_columns),
            feature_columns=list(config.feature_columns),
            noise_mask=np.asarray(selector.noise_mask),
            feature_names=list(data.feature_names),
            selected_features=list(selector.selected_features),
            thresholds={
                name.lower(): float(result["threshold"]) for name, result in results.items()
            },
            sequence_length=config.sequence_length,
        )

    def save(self, models_dir: Path) -> Path:
        """Serializa el artefacto junto a los modelos entrenados."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / ARTIFACTS_FILENAME
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        logger.info(f"Artefactos de inferencia guardados en {path}")
        return path

    @classmethod
    def load(cls, models_dir: Path) -> "ServingArtifacts":
        """Carga el artefacto generado por el ultimo entrenamiento."""
        path = Path(models_dir) / ARTIFACTS_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Ejecute 'fraude train' para generar los "
                "artefactos de inferencia."
            )
        with path.open("rb") as handle:
            return pickle.load(handle)

    def threshold_for(self, kind: ModelKind, default: float = 0.5) -> float:
        """Umbral de decision optimizado en validacion para ese modelo."""
        return float(self.thresholds.get(kind, default))

    def build_matrix(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Convierte transacciones crudas en la matriz que espera el modelo.

        Replica feature engineering, escalado y filtro de ruido del
        entrenamiento. Las variables de ventana (`Amount_roll_*`,
        `Transaction_frequency`) se calculan sobre el lote recibido, por lo que
        un lote pequeno no reproduce exactamente el contexto de entrenamiento.
        """
        missing = [column for column in RAW_COLUMNS if column not in raw.columns]
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")

        config = ExperimentConfig()
        engineered = FraudFeatureEngineer(config).transform(raw, amount_stats=self.amount_stats)
        if len(engineered) != len(raw):
            raise ValueError(
                f"El feature engineering descarto {len(raw) - len(engineered)} de "
                f"{len(raw)} transacciones por valores no finitos."
            )

        engineered[self.robust_columns] = self.robust_scaler.transform(
            engineered[self.robust_columns]
        )
        engineered[self.standard_columns] = self.standard_scaler.transform(
            engineered[self.standard_columns]
        )

        matrix = engineered[self.feature_columns].to_numpy()[:, self.noise_mask]
        return pd.DataFrame(matrix, columns=self.feature_names)

    def lstm_windows(self, matrix: pd.DataFrame) -> np.ndarray:
        """Arma ventanas deslizantes 3D sobre las features seleccionadas.

        Cada ventana puntua su ultima transaccion, asi que las primeras
        `sequence_length - 1` filas del lote no reciben score.
        """
        values = matrix[self.selected_features].to_numpy()
        length = self.sequence_length
        if len(values) < length:
            raise ValueError(
                f"El modelo LSTM necesita al menos {length} transacciones consecutivas "
                f"para formar una ventana; se recibieron {len(values)}."
            )
        windows = [values[i : i + length] for i in range(len(values) - length + 1)]
        return np.asarray(windows)


@dataclass
class ScoringModel:
    """Normaliza el scoring entre los flavors de MLflow que usa el proyecto."""

    kind: ModelKind
    model: Any

    def score(self, features) -> np.ndarray:
        """Devuelve la probabilidad de fraude por fila."""
        if self.kind == "tabnet":
            return self.model.predict_proba(np.asarray(features))[:, 1]
        if self.kind == "lstm":
            return np.asarray(self.model.predict(features, verbose=0)).ravel()

        output = self.model.predict(features)
        if isinstance(output, pd.DataFrame):
            column = "probability" if "probability" in output.columns else output.columns[0]
            return output[column].to_numpy()
        output = np.asarray(output)
        if output.ndim == 2 and output.shape[1] == 2:
            return output[:, 1]
        return output.ravel()


def score_transactions(
    artifacts: ServingArtifacts, scorer: ScoringModel, raw: pd.DataFrame
) -> tuple[np.ndarray, int]:
    """Puntua un lote crudo y devuelve (probabilidades, offset de alineacion).

    El offset indica cuantas transacciones iniciales quedaron sin score porque
    el modelo necesita una ventana previa.
    """
    matrix = artifacts.build_matrix(raw)
    if scorer.kind == "lstm":
        return scorer.score(artifacts.lstm_windows(matrix)), artifacts.sequence_length - 1
    return scorer.score(matrix), 0
