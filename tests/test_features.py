import numpy as np
import pandas as pd
import pytest

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.features import AmountStats, FraudFeatureEngineer


def sample_dataframe(rows=20):
    values = {
        "Time": np.arange(rows) * 3600,
        "Amount": np.linspace(0, 100, rows),
        "Class": np.tile([0, 1], rows // 2),
    }
    values.update({f"V{i}": np.linspace(-i, i, rows) for i in range(1, 29)})
    return pd.DataFrame(values)


def test_feature_engineer_creates_all_configured_columns():
    config = ExperimentConfig()
    transformed = FraudFeatureEngineer(config).transform(sample_dataframe())
    assert set(config.feature_columns).issubset(transformed.columns)
    assert not transformed[config.feature_columns].isnull().any().any()


def test_feature_engineer_does_not_mutate_input():
    original = sample_dataframe()
    before = original.copy(deep=True)
    FraudFeatureEngineer(ExperimentConfig()).transform(original)
    pd.testing.assert_frame_equal(original, before)


def test_amount_stats_are_computed_from_dataframe():
    dataframe = sample_dataframe()
    stats = AmountStats.from_dataframe(dataframe)

    assert stats.mean == pytest.approx(dataframe["Amount"].mean())
    assert stats.std == pytest.approx(dataframe["Amount"].std(ddof=0))
    assert len(stats.log_bin_edges) == 11
