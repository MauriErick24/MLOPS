"""Evaluación predictiva y económica de los modelos."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from deteccion_fraude.config import ExperimentConfig


class FraudModelEvaluator:
    """Define la política de umbral y el paquete de métricas finales."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def find_best_threshold(
        self, y_true: np.ndarray, scores: np.ndarray
    ) -> tuple[float, float]:
        """Maximiza el ROI usando validación, nunca el conjunto de prueba."""
        best_roi, best_threshold = -np.inf, 0.5
        for threshold in np.arange(0.05, 0.9, 0.002):
            prediction = (scores > threshold).astype(int)
            _, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
            savings = tp * self.config.false_negative_cost
            costs = (
                fn * self.config.false_negative_cost
                + fp * self.config.false_positive_cost
            )
            roi = (savings - costs) / (costs + 1e-10) * 100
            if roi > best_roi:
                best_roi, best_threshold = roi, threshold
        return float(best_threshold), float(best_roi)

    def evaluate(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        model_name: str,
        threshold: float,
    ) -> dict:
        prediction = (scores > threshold).astype(int)
        matrix = confusion_matrix(y_true, prediction)
        tn, fp, fn, tp = matrix.ravel()
        savings = tp * self.config.false_negative_cost
        cost_fn = fn * self.config.false_negative_cost
        cost_fp = fp * self.config.false_positive_cost
        total_cost = cost_fn + cost_fp
        return {
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, prediction),
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_true, scores),
            "pr_auc": average_precision_score(y_true, scores),
            "cm": matrix,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "savings": float(savings),
            "cost_fn": float(cost_fn),
            "cost_fp": float(cost_fp),
            "total_cost": float(total_cost),
            "net_benefit": float(savings - total_cost),
            "roi": float((savings - total_cost) / (total_cost + 1e-10) * 100),
        }

    @staticmethod
    def theoretical_best_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
        """Cota diagnóstica; no constituye un umbral válido de producción."""
        precision, recall, _ = precision_recall_curve(y_true, scores)
        values = 2 * precision * recall / (precision + recall + 1e-10)
        return float(values.max())
