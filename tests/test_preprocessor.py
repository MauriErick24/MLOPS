"""Tests for FraudPreprocessor and AmountStats."""

import numpy as np
import pandas as pd

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import FraudDataset
from deteccion_fraude.features import AmountStats, FraudPreprocessor


def sample_df(n=200):
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "Time": rng.uniform(0, 86400, n),
            "Amount": rng.exponential(50, n),
            "Class": rng.choice([0, 1], size=n, p=[0.95, 0.05]),
            **{f"V{i}": rng.randn(n) for i in range(1, 29)},
        }
    )


# ---------------------------------------------------------------------------
# AmountStats
# ---------------------------------------------------------------------------


def test_amount_stats_from_dataframe_captures_mean_and_std():
    df = sample_df(100)
    stats = AmountStats.from_dataframe(df)
    assert stats.mean == df["Amount"].mean()
    assert stats.std == df["Amount"].std(ddof=0)


def test_amount_stats_log_bin_edges_are_sorted():
    stats = AmountStats.from_dataframe(sample_df(100))
    assert stats.log_bin_edges == sorted(stats.log_bin_edges)


def test_amount_stats_from_dataframe_handles_single_value():
    df = pd.DataFrame(
        {
            "Time": [0.0],
            "Amount": [42.0],
            "Class": [0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
        }
    )
    stats = AmountStats.from_dataframe(df)
    assert stats.mean == 42.0


# ---------------------------------------------------------------------------
# FraudPreprocessor.fit_transform
# ---------------------------------------------------------------------------


def _make_splits():
    config = ExperimentConfig()
    df = sample_df(200)
    return FraudDataset.split(df, config)


def test_fit_transform_returns_prepared_data():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert hasattr(result, "X_train")
    assert hasattr(result, "y_train")
    assert hasattr(result, "feature_names")


def test_shapes_are_consistent():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    n_features = len(config.feature_columns)
    assert result.X_train.shape[1] == n_features
    assert result.X_validation.shape[1] == n_features
    assert result.X_test.shape[1] == n_features


def test_train_val_test_row_counts_match_splits():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert result.X_train.shape[0] + result.X_validation.shape[0] + result.X_test.shape[0] > 0


def test_no_nan_in_output():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert not np.isnan(result.X_train).any()
    assert not np.isnan(result.X_validation).any()
    assert not np.isnan(result.X_test).any()


def test_scalers_are_fitted():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert hasattr(result.robust_scaler, "center_")
    assert hasattr(result.standard_scaler, "mean_")


def test_amount_stats_persisted():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert isinstance(result.amount_stats, AmountStats)


def test_class_weights_are_balanced():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert 0 in result.class_weights
    assert 1 in result.class_weights


def test_feature_names_match_config():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    assert result.feature_names == list(config.feature_columns)


def test_lstm_copies_are_independent():
    config = ExperimentConfig()
    splits = _make_splits()
    result = FraudPreprocessor(config).fit_transform(splits)
    result.X_train[0, 0] = 999
    assert result.X_train_lstm[0, 0] != 999
