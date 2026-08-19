from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from deteccion_fraude.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ExperimentConfig

app = typer.Typer()


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
    X_balanced: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    y_balanced: np.ndarray
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
        self.X_balanced = self.X_balanced[:, noise_mask]

        filtered_names = [n for n, keep in zip(self.feature_names, noise_mask) if keep]
        selected_idx = [filtered_names.index(f) for f in selected_features]

        self.X_train_lstm = self.X_train[:, selected_idx]
        self.X_validation_lstm = self.X_validation[:, selected_idx]
        self.X_test_lstm = self.X_test[:, selected_idx]

        self.feature_names = filtered_names


class FraudDataset:
    """Carga, limpieza y particion del dataset."""

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        """Lee CSV, elimina nulos y duplicados."""
        logger.info(f"Cargando dataset desde {path}")
        df = pd.read_csv(path)
        logger.info(f"Shape original: {df.shape}")
        df = df.dropna().reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Shape tras limpieza: {df.shape}")
        return df

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


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    config = ExperimentConfig()
    df = FraudDataset.load(input_path)
    splits = FraudDataset.split(df, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([splits.df_train, splits.df_val, splits.df_test]).to_csv(output_path, index=False)
    logger.success("Dataset cargado y particionado.")


if __name__ == "__main__":
    app()
