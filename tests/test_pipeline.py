"""Tests for the FraudDetectionPipeline facade.

The heavy models are replaced by stubs so the tests exercise the orchestration
(select_features -> train -> evaluate) and the label-alignment logic without
training LSTM or TabNet.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData


def _load_pipeline_module():
    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "tensorflow",
            "deteccion_fraude.modeling.feature_selection",
            "deteccion_fraude.modeling.lstm",
            "deteccion_fraude.modeling.tabnet",
            "deteccion_fraude.evaluation",
            "deteccion_fraude.tracking",
            "deteccion_fraude.serving",
        )
    }

    stub_torch = ModuleType("torch")
    stub_torch.cuda = ModuleType("torch.cuda")
    stub_torch.cuda.is_available = lambda: False
    stub_torch.cuda.manual_seed_all = lambda seed: None
    stub_torch.manual_seed = lambda seed: None

    stub_tf = ModuleType("tensorflow")
    stub_tf.random = ModuleType("tensorflow.random")
    stub_tf.random.set_seed = lambda seed: None

    class _StubSelectorModule(ModuleType):
        class TabNetFeatureSelector:
            def __init__(self, config, device):
                self.config = config
                self.device = device

            def fit_transform(self, data):
                return data

    class _StubLSTMModule(ModuleType):
        class LSTMDetector:
            def __init__(self, config):
                self.config = config

    class _StubTabNetModule(ModuleType):
        class TabNetDetector:
            def __init__(self, config, device):
                self.config = config
                self.device = device

    class _StubEvaluationModule(ModuleType):
        class FraudModelEvaluator:
            def __init__(self, config):
                self.config = config

    class _StubTrackingModule(ModuleType):
        class MLflowFraudTrainer:
            def __init__(self, config):
                self.config = config

    class _StubServingModule(ModuleType):
        class ServingArtifacts:
            def __init__(self):
                self.saved_path = None

            @classmethod
            def from_training(cls, config, data, selector, results):
                return cls()

            def save(self, models_dir):
                self.saved_path = models_dir / "serving_artifacts.pkl"
                return self.saved_path

    sys.modules["torch"] = stub_torch
    sys.modules["tensorflow"] = stub_tf
    sys.modules["deteccion_fraude.modeling.feature_selection"] = _StubSelectorModule(
        "deteccion_fraude.modeling.feature_selection"
    )
    sys.modules["deteccion_fraude.modeling.lstm"] = _StubLSTMModule(
        "deteccion_fraude.modeling.lstm"
    )
    sys.modules["deteccion_fraude.modeling.tabnet"] = _StubTabNetModule(
        "deteccion_fraude.modeling.tabnet"
    )
    sys.modules["deteccion_fraude.evaluation"] = _StubEvaluationModule(
        "deteccion_fraude.evaluation"
    )
    sys.modules["deteccion_fraude.tracking"] = _StubTrackingModule(
        "deteccion_fraude.tracking"
    )
    sys.modules["deteccion_fraude.serving"] = _StubServingModule("deteccion_fraude.serving")

    module_path = Path(__file__).resolve().parents[1] / "deteccion_fraude" / "modeling" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)
    finally:
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


pipeline_mod = _load_pipeline_module()
FraudDetectionPipeline = pipeline_mod.FraudDetectionPipeline


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


def test_save_serving_artifacts_persists_after_evaluate(pipeline, monkeypatch):
    class _StubArtifacts:
        def __init__(self):
            self.saved_path = None

        def save(self, models_dir):
            self.saved_path = models_dir / "serving_artifacts.pkl"
            return self.saved_path

    stub_artifacts = _StubArtifacts()
    pipeline.selector = _StubSelector()
    pipeline.lstm = _StubLSTM()
    pipeline.tabnet = _StubTabNet()
    pipeline.evaluator = _StubEvaluator()
    pipeline.select_features().train().evaluate()

    monkeypatch.setattr(
        pipeline_mod.ServingArtifacts,
        "from_training",
        lambda config, data, selector, results: stub_artifacts,
    )

    saved_path = pipeline.save_serving_artifacts()

    assert saved_path == pipeline.config.models_dir / "serving_artifacts.pkl"
    assert stub_artifacts.saved_path == saved_path
