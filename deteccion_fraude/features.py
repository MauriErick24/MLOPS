"""Ingeniería de características y preprocesamiento sin fuga de información."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats as st
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import DatasetSplits, PreparedData


class FraudFeatureEngineer:
    """Transformador determinista de variables para una partición."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Crea features dentro del split recibido para evitar contaminación."""
        df = dataframe.copy()
        v_columns = self.config.pca_columns

        df["Amount_log"] = np.log1p(df["Amount"])
        df["Amount_log_cat"] = pd.cut(
            np.log1p(df["Amount"]), bins=10, labels=False
        ).astype(float)
        df["Amount_zscore"] = np.abs(st.zscore(df["Amount"]))

        df["Time_hour"] = (df["Time"] / 3600) % 24
        df["Transaction_frequency"] = df.groupby("Time_hour")[
            "Time_hour"
        ].transform("count")

        amount = df["Amount"]
        df["Amount_roll_mean_5"] = amount.rolling(5, min_periods=1).mean()
        df["Amount_roll_std_5"] = (
            amount.rolling(5, min_periods=1).std().fillna(0)
        )
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
        return df.dropna().reset_index(drop=True)


class FraudPreprocessor:
    """Ajusta scalers y SMOTE preservando la separación train/val/test."""

    def __init__(
        self, config: ExperimentConfig, feature_engineer: FraudFeatureEngineer
    ) -> None:
        self.config = config
        self.feature_engineer = feature_engineer
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()

    def prepare(self, splits: DatasetSplits) -> PreparedData:
        train = self.feature_engineer.transform(splits.train)
        validation = self.feature_engineer.transform(splits.validation)
        test = self.feature_engineer.transform(splits.test)

        robust = [c for c in self.config.robust_columns if c in train.columns]
        standard = [c for c in self.config.standard_columns if c in train.columns]

        train.loc[:, robust] = self.robust_scaler.fit_transform(train[robust])
        validation.loc[:, robust] = self.robust_scaler.transform(validation[robust])
        test.loc[:, robust] = self.robust_scaler.transform(test[robust])

        train.loc[:, standard] = self.standard_scaler.fit_transform(train[standard])
        validation.loc[:, standard] = self.standard_scaler.transform(
            validation[standard]
        )
        test.loc[:, standard] = self.standard_scaler.transform(test[standard])

        feature_names = list(self.config.feature_columns)
        X_train = train[feature_names].to_numpy()
        X_validation = validation[feature_names].to_numpy()
        X_test = test[feature_names].to_numpy()
        y_train = train["Class"].to_numpy()
        y_validation = validation["Class"].to_numpy()
        y_test = test["Class"].to_numpy()

        smote = SMOTE(
            sampling_strategy=self.config.smote_ratio,
            random_state=self.config.random_state,
        )
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )

        return PreparedData(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
            X_balanced=X_balanced,
            y_balanced=y_balanced,
            feature_names=feature_names,
            class_weights=dict(zip(np.unique(y_train), weights)),
            robust_scaler=self.robust_scaler,
            standard_scaler=self.standard_scaler,
        )
