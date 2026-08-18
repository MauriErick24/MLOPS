"""Carga, limpieza y partición del dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deteccion_fraude.config import ExperimentConfig


@dataclass
class DatasetSplits:
    """Particiones tabulares antes del escalado."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass
class PreparedData:
    """Matrices preparadas y artefactos aprendidos exclusivamente en train."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    X_balanced: np.ndarray
    y_balanced: np.ndarray
    feature_names: list[str]
    class_weights: dict[int, float]
    robust_scaler: object
    standard_scaler: object

    def apply_feature_selection(
        self, noise_mask: np.ndarray, selected_features: list[str]
    ) -> None:
        """Mantiene todas las matrices sincronizadas tras seleccionar columnas."""
        self.feature_names = [
            name for name, keep in zip(self.feature_names, noise_mask) if keep
        ]
        self.X_train = self.X_train[:, noise_mask]
        self.X_validation = self.X_validation[:, noise_mask]
        self.X_test = self.X_test[:, noise_mask]
        self.X_balanced = self.X_balanced[:, noise_mask]

        indices = [self.feature_names.index(name) for name in selected_features]
        self.X_train_lstm = self.X_train[:, indices]
        self.X_validation_lstm = self.X_validation[:, indices]
        self.X_test_lstm = self.X_test[:, indices]


class FraudDataset:
    """Repositorio de acceso al CSV y responsable de crear los splits."""

    REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {
        f"V{i}" for i in range(1, 29)
    }

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        dataframe = pd.read_csv(self.config.data_file)
        missing = self.REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
        return dataframe

    @staticmethod
    def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Elimina nulos y duplicados exactos sin modificar el objeto recibido."""
        return dataframe.dropna().drop_duplicates().reset_index(drop=True)

    def split(self, dataframe: pd.DataFrame) -> DatasetSplits:
        """Divide 70/15/15, estratifica y ordena cada partición por tiempo."""
        temporary_size = self.config.validation_size + self.config.test_size
        train, temporary = train_test_split(
            dataframe,
            test_size=temporary_size,
            random_state=self.config.random_state,
            stratify=dataframe["Class"],
        )
        relative_test_size = self.config.test_size / temporary_size
        validation, test = train_test_split(
            temporary,
            test_size=relative_test_size,
            random_state=self.config.random_state,
            stratify=temporary["Class"],
        )

        def chronological(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.sort_values("Time").reset_index(drop=True)

        return DatasetSplits(
            train=chronological(train),
            validation=chronological(validation),
            test=chronological(test),
        )
