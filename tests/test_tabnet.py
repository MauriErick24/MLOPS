"""Smoke tests for TabNetDetector.

Training a real TabNet is expensive, so these tests only cover construction and
that the classifier is built with the hyperparameters declared in the module.
"""

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.modeling.tabnet import TabNetDetector


def test_detector_stores_config_and_device():
    config = ExperimentConfig()
    detector = TabNetDetector(config, "cpu")
    assert detector.config is config
    assert detector.device == "cpu"


def test_build_uses_the_configured_hyperparameters():
    model = TabNetDetector(ExperimentConfig(), "cpu")._build()
    assert model.n_d == 32
    assert model.n_a == 32
    assert model.n_steps == 5
    assert model.gamma == 1.3
    assert model.n_independent == 2
    assert model.n_shared == 2
    assert model.lambda_sparse == 1e-3
    assert model.mask_type == "entmax"


def test_build_propagates_the_requested_device():
    model = TabNetDetector(ExperimentConfig(), "cpu")._build()
    assert model.device_name == "cpu"


def test_build_returns_a_trainable_classifier():
    model = TabNetDetector(ExperimentConfig(), "cpu")._build()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "save_model")
