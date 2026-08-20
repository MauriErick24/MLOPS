"""Selección de variables basada en las máscaras de atención de TabNet."""

import time

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import torch

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData


class TabNetFeatureSelector:
    def __init__(self, config: ExperimentConfig, device: str) -> None:
        self.config = config
        self.device = device

    def _build_model(self) -> TabNetClassifier:
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

    def fit_transform(self, data: PreparedData) -> PreparedData:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data.X_balanced,
            data.y_balanced,
            test_size=0.15,
            random_state=self.config.random_state,
            stratify=data.y_balanced,
        )
        self.model = self._build_model()
        started = time.time()
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=["auc", "logloss"],
            max_epochs=50,
            patience=10,
            batch_size=4096,
            virtual_batch_size=512,
        )
        self.training_time = time.time() - started

        raw_importance = self.model.feature_importances_
        self.noise_mask = raw_importance > 1e-6
        filtered_names = [name for name, keep in zip(data.feature_names, self.noise_mask) if keep]
        importance = raw_importance[self.noise_mask]
        importance = importance / importance.sum()
        self.feature_importance = pd.Series(importance, index=filtered_names).sort_values(
            ascending=False
        )
        self.cumulative_importance = np.cumsum(self.feature_importance.values) * 100
        self.features_for_95 = int(np.searchsorted(self.cumulative_importance, 95) + 1)
        self.n_selected = min(
            max(self.features_for_95, self.config.min_lstm_features),
            len(filtered_names),
        )
        self.selected_features = self.feature_importance.head(self.n_selected).index.tolist()
        data.apply_feature_selection(self.noise_mask, self.selected_features)
        return data
