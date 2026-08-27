"""Tests for the FraudDetectionPipeline facade.

The heavy models are replaced by stubs so the tests exercise the orchestration
(select_features -> train -> evaluate) and the label-alignment logic without
training LSTM or TabNet.
"""

import numpy as np
import pytest

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData
import deteccion_fraude.modeling.pipeline as pipeline_mod
from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline


class _StubTrainer:
    """Replaces MLflowFraudTrainer so construction has no MLflow side effects."""

    def __init__(self, config):
        self.config = config


class _StubSelector:
    def __init__(self):
        self.calls = 0

    def fit_transform(self, data):
        self.calls += 1
        return data


class _StubLSTM:
    def __init__(self):
        self.fit_called = False
        self.save_called = False
        self.training_time = 1.0

    def fit(self, data):
        self.fit_called = True
        return self

    def save(self, path):
        self.save_called = True

    def predict_validation(self):
        return np.zeros(4)

    def predict_test(self):
        return np.zeros(4)


class _StubTabNet:
    def __init__(self):
        self.fit_called = False
        self.save_called = False
        self.training_time = 1.0

    def fit(self, data):
        self.fit_called = True
        return self

    def save(self, path):
        self.save_called = True

    def predict_validation(self, data):
        return np.zeros(8)

    def predict_test(self, data):
        return np.zeros(8)


class _StubEvaluator:
    def find_best_threshold(self, y_true, scores):
        return 0.5, 1.0

    def evaluate(self, y_true, scores, name, threshold):
        return {"model": name, "threshold": threshold, "f1": 0.5, "cm": np.eye(2, dtype=int)}

    def theoretical_best_f1(self, y_true, scores):
        return 0.6


def _prepared_data(n=8):
    x = np.zeros((n, 3))
    y = np.array([0, 1] * (n // 2))
    return PreparedData(
        X_train=x,
        X_validation=x,
        X_test=x,
        X_train_lstm=x,
        X_validation_lstm=x,
        X_test_lstm=x,
        y_train=y,
        y_validation=y,
        y_test=y,
        feature_names=["a", "b", "c"],
        class_weights={0: 1.0, 1: 1.0},
        robust_scaler=None,
        standard_scaler=None,
        amount_stats=None,
    )


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "MLflowFraudTrainer", _StubTrainer)
    config = ExperimentConfig(models_dir=tmp_path, figures_dir=tmp_path)
    return FraudDetectionPipeline(config, _prepared_data())


def test_construction_wires_the_specialized_components(pipeline):
    assert pipeline.selector is not None
    assert pipeline.lstm is not None
    assert pipeline.tabnet is not None
    assert pipeline.evaluator is not None
    assert pipeline.results == {}


def test_save_serving_artifacts_requires_evaluate_first(pipeline):
    with pytest.raises(RuntimeError):
        pipeline.save_serving_artifacts()


def test_pipeline_runs_the_three_stages_in_order(pipeline):
    pipeline.selector = _StubSelector()
    pipeline.lstm = _StubLSTM()
    pipeline.tabnet = _StubTabNet()
    pipeline.evaluator = _StubEvaluator()

    assert pipeline.select_features() is pipeline
    assert pipeline.train() is pipeline
    results = pipeline.evaluate()

    assert pipeline.selector.calls == 1
    assert pipeline.lstm.fit_called and pipeline.lstm.save_called
    assert pipeline.tabnet.fit_called and pipeline.tabnet.save_called
    assert set(results) == {"LSTM", "TabNet"}
    assert all("theoretical_best_f1" in result for result in results.values())


def test_evaluate_aligns_labels_by_the_lstm_window(pipeline):
    pipeline.selector = _StubSelector()
    pipeline.lstm = _StubLSTM()
    pipeline.tabnet = _StubTabNet()
    pipeline.evaluator = _StubEvaluator()

    pipeline.select_features().train().evaluate()

    # The first sequence_length - 1 rows have no score, so the aligned test
    # labels drop exactly that many entries.
    offset = pipeline.config.sequence_length - 1
    assert len(pipeline.y_test_aligned) == len(pipeline.data.y_test) - offset
