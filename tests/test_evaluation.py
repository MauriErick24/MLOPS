import numpy as np
import pytest

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


def test_evaluate_computes_cost_and_roi_fields():
    evaluator = FraudModelEvaluator(ExperimentConfig())
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = evaluator.evaluate(y_true, scores, "modelo", threshold=0.5)

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 2
    assert result["total_cost"] == 0.0
    assert result["net_benefit"] == result["savings"]
    assert result["roi"] > 0


def test_theoretical_best_f1_is_one_for_perfect_ranking():
    evaluator = FraudModelEvaluator(ExperimentConfig())
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert evaluator.theoretical_best_f1(y_true, scores) == pytest.approx(1.0)
