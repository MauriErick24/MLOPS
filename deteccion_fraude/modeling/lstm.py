"""Modelo LSTM para ventanas de transacciones."""

from pathlib import Path
import time

from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.dataset import PreparedData


def _focal_loss_fn(gamma: float = 2.0, alpha: float = 0.75):
    """Closure compatible con Keras 3: retorna loss escalar por batch."""

    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * K.pow(1.0 - pt, gamma)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        return K.mean(focal_weight * cross_entropy)

    return focal_loss


class LSTMDetector:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @staticmethod
    def create_sequences(
        data: np.ndarray, labels: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sequences = [data[i : i + length] for i in range(len(data) - length + 1)]
        targets = [labels[i + length - 1] for i in range(len(data) - length + 1)]
        return np.asarray(sequences), np.asarray(targets)

    def _build(self, n_features: int) -> tf.keras.Model:
        sequence_length = self.config.sequence_length
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(sequence_length, n_features)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=_focal_loss_fn(gamma=2.0, alpha=0.75),
            metrics=[
                "accuracy",
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        return model

    def fit(self, data: PreparedData) -> "LSTMDetector":
        from loguru import logger

        length = self.config.sequence_length
        train_X, train_y = self.create_sequences(data.X_train_lstm, data.y_train, length)
        logger.info(f"LSTM train shape: {train_X.shape}, class distribution: {dict(zip(*np.unique(train_y, return_counts=True)))}")
        logger.info(f"LSTM class_weights: {data.class_weights}")
        rows, steps, features = train_X.shape
        flattened = train_X.reshape(rows, steps * features)
        smote = SMOTE(
            sampling_strategy=self.config.smote_ratio,
            random_state=self.config.random_state,
        )
        balanced_X, balanced_y = smote.fit_resample(flattened, train_y)
        balanced_X = balanced_X.reshape(-1, steps, features)

        self.validation_X, self.validation_y = self.create_sequences(
            data.X_validation_lstm, data.y_validation, length
        )
        self.test_X, self.test_y = self.create_sequences(data.X_test_lstm, data.y_test, length)
        self.model = self._build(features)
        started = time.time()
        self.history = self.model.fit(
            balanced_X,
            balanced_y,
            validation_data=(self.validation_X, self.validation_y),
            epochs=100,
            batch_size=4096,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ],
            verbose=0,
        )
        self.training_time = time.time() - started
        final_loss = self.history.history["loss"][-1]
        final_val_loss = self.history.history["val_loss"][-1]
        logger.info(f"LSTM training: {len(self.history.epoch)} epochs, loss={final_loss:.4f}, val_loss={final_val_loss:.4f}")
        return self

    def predict_validation(self) -> np.ndarray:
        from loguru import logger

        preds = self.model.predict(self.validation_X, verbose=0).ravel()
        logger.info(f"LSTM val predictions: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}, >0.5 count={int((preds > 0.5).sum())}/{len(preds)}")
        return preds

    def predict_test(self) -> np.ndarray:
        return self.model.predict(self.test_X, verbose=0).ravel()

    def save(self, path: Path) -> None:
        self.model.save(path)
