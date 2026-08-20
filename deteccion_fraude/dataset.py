from dataclasses import dataclass

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deteccion_fraude.config import ExperimentConfig


@dataclass
class DatasetSplits:
    """Splits crudos antes de feature engineering."""

    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame


@dataclass
class PreparedData:
    """Matrices numericas listas para los modelos."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    X_train_lstm: np.ndarray
    X_validation_lstm: np.ndarray
    X_test_lstm: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_weights: dict[int, float]
    robust_scaler: object
    standard_scaler: object

    def apply_feature_selection(
        self, noise_mask: np.ndarray, selected_features: list[str]
    ) -> None:
        """Filtra features de ruido y subselecciona para LSTM. Mutacion in-place."""
        self.X_train = self.X_train[:, noise_mask]
        self.X_validation = self.X_validation[:, noise_mask]
        self.X_test = self.X_test[:, noise_mask]

        filtered_names = [n for n, keep in zip(self.feature_names, noise_mask) if keep]
        selected_idx = [filtered_names.index(f) for f in selected_features]

        self.X_train_lstm = self.X_train[:, selected_idx]
        self.X_validation_lstm = self.X_validation[:, selected_idx]
        self.X_test_lstm = self.X_test[:, selected_idx]

        self.feature_names = filtered_names


class FraudDataset:
    """Carga, limpieza y particion del dataset."""

    REQUIRED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}

    @staticmethod
    def load(path) -> pd.DataFrame:
        """Lee CSV, elimina nulos y duplicados."""
        logger.info(f"Cargando dataset desde {path}")
        df = pd.read_csv(path)
        missing = FraudDataset.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
        logger.info(f"Shape original: {df.shape}")
        df = FraudDataset.clean(df)
        logger.info(f"Shape tras limpieza: {df.shape}")
        return df

    @staticmethod
    def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Elimina nulos y duplicados exactos sin modificar el input."""
        return dataframe.dropna().drop_duplicates().reset_index(drop=True)

    @staticmethod
    def split(df: pd.DataFrame, config: ExperimentConfig) -> DatasetSplits:
        """Particion estratificada 70/15/15 ordenada por Time."""
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=config.random_state, stratify=df["Class"]
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.5, random_state=config.random_state, stratify=df_temp["Class"]
        )
        df_train = df_train.sort_values("Time").reset_index(drop=True)
        df_val = df_val.sort_values("Time").reset_index(drop=True)
        df_test = df_test.sort_values("Time").reset_index(drop=True)
        logger.info(f"Splits — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
        return DatasetSplits(df_train=df_train, df_val=df_val, df_test=df_test)
