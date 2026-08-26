import numpy as np
import pandas as pd
import pytest

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import FraudDataset


def sample_df(n=200):
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "Time": rng.uniform(0, 86400, n),
            "Amount": rng.exponential(50, n),
            "Class": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            **{f"V{i}": rng.randn(n) for i in range(1, 29)},
        }
    )
    return df


def test_load_drops_duplicates(tmp_path):
    df = sample_df(100)
    pd.concat([df, df]).to_csv(tmp_path / "dup.csv", index=False)
    result = FraudDataset.load(tmp_path / "dup.csv")
    assert len(result) == 100


def test_split_counts_sum_to_total():
    df = sample_df(300)
    config = ExperimentConfig()
    splits = FraudDataset.split(df, config)
    total = len(splits.df_train) + len(splits.df_val) + len(splits.df_test)
    assert total == len(df)


def test_split_preserves_class_proportion():
    df = sample_df(500)
    config = ExperimentConfig()
    splits = FraudDataset.split(df, config)
    orig_rate = df["Class"].mean()
    for part in (splits.df_train, splits.df_val, splits.df_test):
        assert abs(part["Class"].mean() - orig_rate) < 0.05


def test_split_sorted_by_time():
    df = sample_df(200)
    config = ExperimentConfig()
    splits = FraudDataset.split(df, config)
    for part in (splits.df_train, splits.df_val, splits.df_test):
        assert list(part["Time"]) == sorted(part["Time"])


def test_config_reads_mlflow_tracking_uri_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert ExperimentConfig().mlflow_tracking_uri == "http://mlflow:5000"
