import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.features import AmountStats, FraudFeatureEngineer
from deteccion_fraude.serving import ScoringModel, ServingArtifacts, score_transactions


def raw_transactions(rows=40, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "Time": np.sort(rng.uniform(0, 86400, rows)),
            **{f"V{i}": rng.randn(rows) for i in range(1, 29)},
            "Amount": rng.exponential(50, rows),
        }
    )


def build_artifacts(config=None, n_selected=10):
    """Artefactos equivalentes a los que produce un entrenamiento real."""
    config = config or ExperimentConfig()
    training = raw_transactions(rows=200, seed=7)
    amount_stats = AmountStats.from_dataframe(training)
    engineered = FraudFeatureEngineer(config).transform(training)

    robust_scaler = RobustScaler().fit(engineered[config.robust_columns])
    standard_scaler = StandardScaler().fit(engineered[config.standard_columns])

    feature_columns = list(config.feature_columns)
    return ServingArtifacts(
        robust_scaler=robust_scaler,
        standard_scaler=standard_scaler,
        amount_stats=amount_stats,
        robust_columns=list(config.robust_columns),
        standard_columns=list(config.standard_columns),
        feature_columns=feature_columns,
        noise_mask=np.ones(len(feature_columns), dtype=bool),
        feature_names=feature_columns,
        selected_features=feature_columns[:n_selected],
        thresholds={"tabnet": 0.42, "lstm": 0.31},
        sequence_length=config.sequence_length,
    )


class StubTabNet:
    """Modelo de dos clases con probabilidad creciente por fila."""

    def predict_proba(self, matrix):
        fraud = np.linspace(0.1, 0.9, len(matrix))
        return np.column_stack([1 - fraud, fraud])


class StubLSTM:
    def predict(self, windows, verbose=0):
        return np.full((len(windows), 1), 0.7)


def test_build_matrix_returns_selected_feature_columns():
    artifacts = build_artifacts()
    matrix = artifacts.build_matrix(raw_transactions(rows=30))
    assert list(matrix.columns) == artifacts.feature_names
    assert len(matrix) == 30
    assert np.isfinite(matrix.to_numpy()).all()


def test_build_matrix_scores_a_single_transaction():
    """Sin estadisticos persistidos el z-score de una fila es NaN y se descarta."""
    artifacts = build_artifacts()
    matrix = artifacts.build_matrix(raw_transactions(rows=1))
    assert len(matrix) == 1
    assert np.isfinite(matrix.to_numpy()).all()


def test_build_matrix_rejects_missing_columns():
    artifacts = build_artifacts()
    incomplete = raw_transactions(rows=5).drop(columns=["V7", "Amount"])
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        artifacts.build_matrix(incomplete)


def test_amount_stats_are_independent_of_batch_size():
    artifacts = build_artifacts()
    batch = raw_transactions(rows=12, seed=3)
    full = artifacts.build_matrix(batch)
    single = artifacts.build_matrix(batch.iloc[[4]].reset_index(drop=True))
    assert single["Amount_zscore"].iloc[0] == pytest.approx(full["Amount_zscore"].iloc[4])


def test_lstm_windows_shape_matches_sequence_length():
    artifacts = build_artifacts()
    matrix = artifacts.build_matrix(raw_transactions(rows=20))
    windows = artifacts.lstm_windows(matrix)
    assert windows.shape == (
        20 - artifacts.sequence_length + 1,
        artifacts.sequence_length,
        len(artifacts.selected_features),
    )


def test_lstm_windows_requires_enough_transactions():
    artifacts = build_artifacts()
    matrix = artifacts.build_matrix(raw_transactions(rows=3))
    with pytest.raises(ValueError, match="al menos"):
        artifacts.lstm_windows(matrix)


def test_score_transactions_tabnet_has_no_offset():
    artifacts = build_artifacts()
    scorer = ScoringModel(kind="tabnet", model=StubTabNet())
    probabilities, offset = score_transactions(artifacts, scorer, raw_transactions(rows=15))
    assert offset == 0
    assert len(probabilities) == 15


def test_score_transactions_lstm_offsets_by_window():
    artifacts = build_artifacts()
    scorer = ScoringModel(kind="lstm", model=StubLSTM())
    probabilities, offset = score_transactions(artifacts, scorer, raw_transactions(rows=15))
    assert offset == artifacts.sequence_length - 1
    assert len(probabilities) == 15 - offset


def test_threshold_falls_back_when_model_missing():
    artifacts = build_artifacts()
    assert artifacts.threshold_for("tabnet") == pytest.approx(0.42)
    assert artifacts.threshold_for("pyfunc") == pytest.approx(0.5)


def test_artifacts_roundtrip_through_disk(tmp_path):
    artifacts = build_artifacts()
    artifacts.save(tmp_path)
    restored = ServingArtifacts.load(tmp_path)
    assert restored.feature_names == artifacts.feature_names
    assert restored.thresholds == artifacts.thresholds
    pd.testing.assert_frame_equal(
        restored.build_matrix(raw_transactions(rows=8)),
        artifacts.build_matrix(raw_transactions(rows=8)),
    )


def test_load_without_training_explains_next_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="fraude train"):
        ServingArtifacts.load(tmp_path)
