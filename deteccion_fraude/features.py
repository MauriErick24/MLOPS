from typing import TYPE_CHECKING

from loguru import logger
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import typer

from deteccion_fraude.config import ExperimentConfig

if TYPE_CHECKING:
    from deteccion_fraude.dataset import DatasetSplits, PreparedData

app = typer.Typer()


class FraudFeatureEngineer:
    """Crea variables derivadas a partir del dataset crudo."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Aplica feature engineering sin mutar el input."""
        df = dataframe.copy()
        v_columns = self.config.pca_columns

        df["Amount_log"] = np.log1p(df["Amount"])
        df["Amount_log_cat"] = pd.cut(np.log1p(df["Amount"]), bins=10, labels=False).astype(float)
        df["Amount_zscore"] = np.abs(st.zscore(df["Amount"]))

        df["Time_hour"] = (df["Time"] / 3600) % 24
        df["Transaction_frequency"] = df.groupby("Time_hour")["Time_hour"].transform("count")

        amount = df["Amount"]
        df["Amount_roll_mean_5"] = amount.rolling(5, min_periods=1).mean()
        df["Amount_roll_std_5"] = amount.rolling(5, min_periods=1).std().fillna(0)
        df["Amount_roll_mean_10"] = amount.rolling(10, min_periods=1).mean()
        df["Amount_roll_ratio"] = amount / (df["Amount_roll_mean_5"] + 1e-6)

        df["V_mean"] = df[v_columns].mean(axis=1)
        df["V_std"] = df[v_columns].std(axis=1)
        df["V_max"] = df[v_columns].max(axis=1)
        df["V_min"] = df[v_columns].min(axis=1)
        df["V_median"] = df[v_columns].median(axis=1)
        df["V_abs_sum"] = df[v_columns].abs().sum(axis=1)
        df["V_count_neg"] = (df[v_columns] < 0).sum(axis=1)
        df["V_count_pos"] = (df[v_columns] > 0).sum(axis=1)

        df["V1_V2_ratio"] = df["V1"] / (df["V2"].abs() + 1e-6)
        df["V3_V4_ratio"] = df["V3"] / (df["V4"].abs() + 1e-6)
        df["V12_V14_ratio"] = df["V12"] / (df["V14"].abs() + 1e-6)

        df["Amount_V_ratio"] = amount / (np.abs(df["V_mean"]) + 1e-6)
        df["Amount_V_std_ratio"] = amount / (df["V_std"] + 1e-6)

        df = df.dropna().reset_index(drop=True)
        return df


class FraudPreprocessor:
    """Escalado y preparacion de matrices para modelos."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.engineer = FraudFeatureEngineer(config)

    def fit_transform(self, splits: "DatasetSplits") -> "PreparedData":
        """Transforma splits en PreparedData con scaling."""
        from deteccion_fraude.dataset import PreparedData

        df_train = self.engineer.transform(splits.df_train)
        df_val = self.engineer.transform(splits.df_val)
        df_test = self.engineer.transform(splits.df_test)

        robust_scaler = RobustScaler()
        standard_scaler = StandardScaler()

        robust = self.config.robust_columns
        standard = self.config.standard_columns

        df_train[robust] = robust_scaler.fit_transform(df_train[robust])
        df_train[standard] = standard_scaler.fit_transform(df_train[standard])
        df_val[robust] = robust_scaler.transform(df_val[robust])
        df_val[standard] = standard_scaler.transform(df_val[standard])
        df_test[robust] = robust_scaler.transform(df_test[robust])
        df_test[standard] = standard_scaler.transform(df_test[standard])

        feature_cols = self.config.feature_columns
        X_train = df_train[feature_cols].values
        X_val = df_val[feature_cols].values
        X_test = df_test[feature_cols].values
        y_train = df_train["Class"].values
        y_val = df_val["Class"].values
        y_test = df_test["Class"].values

        weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train).astype(int), weights))

        logger.info(
            f"PreparedData — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
        )

        return PreparedData(
            X_train=X_train,
            X_validation=X_val,
            X_test=X_test,
            X_train_lstm=X_train.copy(),
            X_validation_lstm=X_val.copy(),
            X_test_lstm=X_test.copy(),
            y_train=y_train,
            y_validation=y_val,
            y_test=y_test,
            feature_names=list(feature_cols),
            class_weights=class_weights,
            robust_scaler=robust_scaler,
            standard_scaler=standard_scaler,
        )
