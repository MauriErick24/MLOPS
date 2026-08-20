"""Modelo TabNet final."""

from pathlib import Path
import time

from imblearn.over_sampling import SMOTE
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData


class TabNetDetector:
    def __init__(self, config: ExperimentConfig, device: str) -> None:
        self.config = config
        self.device = device

    def _build(self) -> TabNetClassifier:
        return TabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 1e-2},
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            verbose=0,
            device_name=self.device,
        )

    def fit(self, data: PreparedData) -> "TabNetDetector":
        self.model = self._build()
        smote = SMOTE(
            sampling_strategy=self.config.smote_ratio,
            random_state=self.config.random_state,
        )
        X_balanced, y_balanced = smote.fit_resample(data.X_train, data.y_train)
        started = time.time()
        self.model.fit(
            X_balanced,
            y_balanced,
            eval_set=[(data.X_validation, data.y_validation)],
            eval_metric=["auc", "logloss"],
            max_epochs=50,
            patience=10,
            batch_size=4096,
            virtual_batch_size=512,
        )
        self.training_time = time.time() - started
        return self

    def predict_validation(self, data: PreparedData) -> np.ndarray:
        return self.model.predict_proba(data.X_validation)[:, 1]

    def predict_test(self, data: PreparedData) -> np.ndarray:
        return self.model.predict_proba(data.X_test)[:, 1]

    def save(self, path_without_extension: Path) -> None:
        self.model.save_model(str(path_without_extension))
