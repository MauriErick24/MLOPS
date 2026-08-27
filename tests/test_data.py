import numpy as np
import pandas as pd
import pytest

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import FraudDataset, PreparedData


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


def test_load_raises_when_required_columns_are_missing(tmp_path):
    df = sample_df(20).drop(columns=["Amount", "V7"])
    csv_path = tmp_path / "broken.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        FraudDataset.load(csv_path)


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


def test_apply_feature_selection_keeps_matrices_and_names_in_sync():
    data = PreparedData(
        X_train=np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]),
        X_validation=np.array([[110.0, 111.0, 112.0, 113.0]]),
        X_test=np.array([[210.0, 211.0, 212.0, 213.0]]),
        X_train_lstm=np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]),
        X_validation_lstm=np.array([[110.0, 111.0, 112.0, 113.0]]),
        X_test_lstm=np.array([[210.0, 211.0, 212.0, 213.0]]),
        y_train=np.array([0, 1]),
        y_validation=np.array([0]),
        y_test=np.array([1]),
        feature_names=["f1", "f2", "f3", "f4"],
        class_weights={0: 1.0, 1: 1.0},
        robust_scaler=None,
        standard_scaler=None,
        amount_stats=None,
    )

    noise_mask = np.array([True, False, True, True])
    selected_features = ["f4", "f1"]

    data.apply_feature_selection(noise_mask, selected_features)

    assert data.feature_names == ["f1", "f3", "f4"]
    np.testing.assert_array_equal(
        data.X_train,
        np.array([[10.0, 12.0, 13.0], [20.0, 22.0, 23.0]]),
    )
    np.testing.assert_array_equal(data.X_validation, np.array([[110.0, 112.0, 113.0]]))
    np.testing.assert_array_equal(data.X_test, np.array([[210.0, 212.0, 213.0]]))
    np.testing.assert_array_equal(
        data.X_train_lstm,
        np.array([[13.0, 10.0], [23.0, 20.0]]),
    )
    np.testing.assert_array_equal(data.X_validation_lstm, np.array([[113.0, 110.0]]))
    np.testing.assert_array_equal(data.X_test_lstm, np.array([[213.0, 210.0]]))


def test_config_reads_mlflow_tracking_uri_from_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert ExperimentConfig().mlflow_tracking_uri == "http://mlflow:5000"
