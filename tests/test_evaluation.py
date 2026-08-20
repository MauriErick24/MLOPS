import numpy as np

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.evaluation import FraudModelEvaluator


def test_evaluator_returns_complete_confusion_matrix():
    evaluator = FraudModelEvaluator(ExperimentConfig())
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    result = evaluator.evaluate(y_true, scores, "modelo", threshold=0.5)
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == 4
    assert result["f1"] == 1.0


def test_threshold_is_inside_search_interval():
    evaluator = FraudModelEvaluator(ExperimentConfig())
    threshold, _ = evaluator.find_best_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.7, 0.9])
    )
    assert 0.05 <= threshold < 0.9
