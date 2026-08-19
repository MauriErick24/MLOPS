from pathlib import Path
from typing import TYPE_CHECKING

from imblearn.over_sampling import SMOTE
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import typer

from deteccion_fraude.config import PROCESSED_DATA_DIR, ExperimentConfig

if TYPE_CHECKING:
    from deteccion_fraude.dataset import DatasetSplits, PreparedData

app = typer.Typer()


class FraudFeatureEngineer:
    """Crea variables derivadas a partir del dataset crudo."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica feature engineering sin mutar el input."""
        df = df.copy()
        v_cols = [f"V{i}" for i in range(1, 29)]

        df["Amount_log"] = np.log1p(df["Amount"])
        df["Amount_log_cat"] = pd.cut(np.log1p(df["Amount"]), bins=10, labels=False).astype(float)
        df["Amount_zscore"] = np.abs(df["Amount"] - df["Amount"].mean()) / (
            df["Amount"].std() + 1e-6
        )

        df["Time_hour"] = (df["Time"] / 3600) % 24
        df["Transaction_frequency"] = df.groupby("Time_hour")["Time_hour"].transform("count")

        df["Amount_roll_mean_5"] = df["Amount"].rolling(window=5, min_periods=1).mean()
        df["Amount_roll_std_5"] = df["Amount"].rolling(window=5, min_periods=1).std().fillna(0)
        df["Amount_roll_mean_10"] = df["Amount"].rolling(window=10, min_periods=1).mean()
        df["Amount_roll_ratio"] = df["Amount"] / (df["Amount_roll_mean_5"] + 1e-6)

        df["V_mean"] = df[v_cols].mean(axis=1)
        df["V_std"] = df[v_cols].std(axis=1)
        df["V_max"] = df[v_cols].max(axis=1)
        df["V_min"] = df[v_cols].min(axis=1)
        df["V_median"] = df[v_cols].median(axis=1)
        df["V_abs_sum"] = df[v_cols].abs().sum(axis=1)
        df["V_count_neg"] = (df[v_cols] < 0).sum(axis=1)
        df["V_count_pos"] = (df[v_cols] > 0).sum(axis=1)

        df["V1_V2_ratio"] = df["V1"] / (df["V2"].abs() + 1e-6)
        df["V3_V4_ratio"] = df["V3"] / (df["V4"].abs() + 1e-6)
        df["V12_V14_ratio"] = df["V12"] / (df["V14"].abs() + 1e-6)

        df["Amount_V_ratio"] = df["Amount"] / (df["V_mean"].abs() + 1e-6)
        df["Amount_V_std_ratio"] = df["Amount"] / (df["V_std"] + 1e-6)

        df = df.dropna().reset_index(drop=True)
        return df


class FraudPreprocessor:
    """Escalado, SMOTE y preparacion de matrices para modelos."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.engineer = FraudFeatureEngineer(config)

    def fit_transform(self, splits: "DatasetSplits") -> "PreparedData":
        """Transforma splits en PreparedData con scaling y SMOTE."""
        from deteccion_fraude.dataset import PreparedData

        df_train = self.engineer.transform(splits.df_train)
        df_val = self.engineer.transform(splits.df_val)
        df_test = self.engineer.transform(splits.df_test)

        robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()

        df_train[self.config.robust_columns] = robust_scaler.fit_transform(
            df_train[self.config.robust_columns]
        )
        df_train[self.config.standard_columns] = standard_scaler.fit_transform(
            df_train[self.config.standard_columns]
        )
        df_val[self.config.robust_columns] = robust_scaler.transform(
            df_val[self.config.robust_columns]
        )
        df_val[self.config.standard_columns] = standard_scaler.transform(
            df_val[self.config.standard_columns]
        )
        df_test[self.config.robust_columns] = robust_scaler.transform(
            df_test[self.config.robust_columns]
        )
        df_test[self.config.standard_columns] = standard_scaler.transform(
            df_test[self.config.standard_columns]
        )

        feature_cols = self.config.feature_columns
        X_train = df_train[feature_cols].values
        X_val = df_val[feature_cols].values
        X_test = df_test[feature_cols].values
        y_train = df_train["Class"].values
        y_val = df_val["Class"].values
        y_test = df_test["Class"].values

        smote = SMOTE(
            sampling_strategy=self.config.smote_ratio,
            random_state=self.config.random_state,
        )
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

        unique, counts = np.unique(y_train, return_counts=True)
        class_weights = dict(zip(unique.astype(int), (1.0 / counts) * counts.sum() / 2.0))

        logger.info(
            f"PreparedData — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
        )
        logger.info(f"SMOTE — balanced: {X_balanced.shape}")

        return PreparedData(
            X_train=X_train,
            X_validation=X_val,
            X_test=X_test,
            X_train_lstm=X_train.copy(),
            X_validation_lstm=X_val.copy(),
            X_test_lstm=X_test.copy(),
            X_balanced=X_balanced,
            y_train=y_train,
            y_validation=y_val,
            y_test=y_test,
            y_balanced=y_balanced,
            feature_names=list(feature_cols),
            class_weights=class_weights,
            robust_scaler=robust_scaler,
            standard_scaler=standard_scaler,
        )


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    from deteccion_fraude.dataset import FraudDataset

    config = ExperimentConfig()
    df = FraudDataset.load(input_path)
    splits = FraudDataset.split(df, config)
    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform(splits)
    logger.success(f"Features generadas: {len(data.feature_names)} columnas")


if __name__ == "__main__":
    app()
