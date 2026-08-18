"""Fachada orientada a objetos para el experimento completo."""

import pickle

import numpy as np
import tensorflow as tf
import torch

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData
from deteccion_fraude.evaluation import FraudModelEvaluator
from deteccion_fraude.modeling.feature_selection import TabNetFeatureSelector
from deteccion_fraude.modeling.lstm import LSTMDetector
from deteccion_fraude.modeling.tabnet import TabNetDetector


class FraudDetectionPipeline:
    """Coordina componentes especializados mediante composición."""

    def __init__(self, config: ExperimentConfig, data: PreparedData) -> None:
        self.config = config
        self.data = data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.selector = TabNetFeatureSelector(config, self.device)
        self.lstm = LSTMDetector(config)
        self.tabnet = TabNetDetector(config, self.device)
        self.evaluator = FraudModelEvaluator(config)
        self.results: dict[str, dict] = {}

        np.random.seed(config.random_state)
        tf.random.set_seed(config.random_state)
        torch.manual_seed(config.random_state)
        torch.cuda.manual_seed_all(config.random_state)
        config.create_output_directories()

    def select_features(self) -> "FraudDetectionPipeline":
        self.selector.fit_transform(self.data)
        return self

    def train(self) -> "FraudDetectionPipeline":
        self.lstm.fit(self.data)
        self.tabnet.fit(self.data)
        self.lstm.save(self.config.models_dir / "lstm_fraud_detector.keras")
        self.tabnet.save(self.config.models_dir / "tabnet_fraud_detector")
        return self

    def evaluate(self) -> dict[str, dict]:
        offset = self.config.sequence_length - 1
        y_validation = self.data.y_validation[offset:]
        y_test = self.data.y_test[offset:]
        self.validation_scores = {
            "LSTM": self.lstm.predict_validation(),
            "TabNet": self.tabnet.predict_validation(self.data)[offset:],
        }
        self.test_scores = {
            "LSTM": self.lstm.predict_test(),
            "TabNet": self.tabnet.predict_test(self.data)[offset:],
        }

        for name in ("LSTM", "TabNet"):
            threshold, _ = self.evaluator.find_best_threshold(
                y_validation, self.validation_scores[name]
            )
            result = self.evaluator.evaluate(
                y_test, self.test_scores[name], name, threshold
            )
            result["theoretical_best_f1"] = self.evaluator.theoretical_best_f1(
                y_test, self.test_scores[name]
            )
            self.results[name] = result
        self.y_test_aligned = y_test
        return self.results

    def save_artifacts(self) -> None:
        clean_results = {
            name.lower(): {key: value for key, value in result.items() if key != "cm"}
            for name, result in self.results.items()
        }
        payload = {
            **clean_results,
            "selected_features": self.selector.selected_features,
            "n_features_lstm": self.selector.n_selected,
            "n_features_tabnet": len(self.data.feature_names),
            "times": {
                "lstm": self.lstm.training_time,
                "tabnet": self.tabnet.training_time,
            },
        }
        with (self.config.models_dir / "results.pkl").open("wb") as file:
            pickle.dump(payload, file)
        with (self.config.models_dir / "scaler.pkl").open("wb") as file:
            pickle.dump(
                {
                    "robust": self.data.robust_scaler,
                    "standard": self.data.standard_scaler,
                },
                file,
            )
