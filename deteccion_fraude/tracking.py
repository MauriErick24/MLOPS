"""MLflow tracking, lineage y serving para deteccion de fraude."""

import subprocess

from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
import mlflow.sklearn
import pandas as pd

from deteccion_fraude.config import ExperimentConfig


def get_lineage_metadata() -> dict[str, str]:
    """Extrae hash de Git y estado de DVC para trazabilidad de auditoria."""
    metadata: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["git_commit"] = result.stdout.strip() or "standalone"
    except OSError:
        metadata["git_commit"] = "git_unavailable"

    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["dvc_status"] = "synced" if not result.stdout.strip() else "uncommitted"
    except OSError:
        metadata["dvc_status"] = "dvc_unavailable"

    return metadata


class FraudDecisionWrapper(PythonModel):
    """Wrapper PyFunc con umbral de decision y flag de riesgo alto."""

    def __init__(self, trained_model, decision_threshold: float = 0.5):
        self.model = trained_model
        self.decision_threshold = decision_threshold

    def predict(self, context, model_input):
        probabilities = self.model.predict_proba(model_input)[:, 1]
        decisions = (probabilities >= self.decision_threshold).astype(int)
        return pd.DataFrame(
            {
                "probability": probabilities,
                "prediction": decisions,
                "high_risk_flag": (probabilities >= 0.85).astype(int),
            }
        )


class MLflowFraudTrainer:
    """Orquesta training con MLflow Tracking para ambos modelos."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

    def log_training(
        self,
        lstm_detector,
        tabnet_detector,
        data,
        results: dict,
    ) -> str:
        """Registra un run completo con ambos modelos en MLflow."""
        lineage = get_lineage_metadata()
        params = {
            "sequence_length": self.config.sequence_length,
            "smote_ratio": self.config.smote_ratio,
            "random_state": self.config.random_state,
            "false_negative_cost": self.config.false_negative_cost,
            "false_positive_cost": self.config.false_positive_cost,
        }

        with mlflow.start_run(run_name="fraud_detection_pipeline") as run:
            mlflow.set_tags(
                {
                    "environment": "development",
                    "architecture": "oop_modular",
                    "owner": "grupo_2",
                    **lineage,
                }
            )
            mlflow.log_params(params)

            for name, result in results.items():
                mlflow.log_metric(f"{name.lower()}_f1", result["f1"])
                mlflow.log_metric(f"{name.lower()}_precision", result["precision"])
                mlflow.log_metric(f"{name.lower()}_recall", result["recall"])
                mlflow.log_metric(f"{name.lower()}_roc_auc", result["roc_auc"])
                mlflow.log_metric(f"{name.lower()}_roi", result["roi"])
                mlflow.log_metric(f"{name.lower()}_threshold", result["threshold"])

            mlflow.log_metric("lstm_training_time", lstm_detector.training_time)
            mlflow.log_metric("tabnet_training_time", tabnet_detector.training_time)

            if lstm_detector.model is not None:
                X_sample = data.X_validation_lstm[:5]
                try:
                    signature = infer_signature(
                        X_sample, lstm_detector.model.predict(X_sample)
                    )
                except (ValueError, TypeError):
                    signature = None
                mlflow.keras.log_model(
                    lstm_detector.model,
                    artifact_path="lstm_model",
                    signature=signature,
                )

            if tabnet_detector.model is not None:
                X_sample = data.X_validation[:5]
                try:
                    signature = infer_signature(
                        X_sample, tabnet_detector.model.predict(X_sample)
                    )
                except (ValueError, TypeError):
                    signature = None
                mlflow.sklearn.log_model(
                    tabnet_detector.model,
                    artifact_path="tabnet_model",
                    signature=signature,
                    input_example=X_sample[:3],
                )

            for artifact_path in (
                self.config.figures_dir / "overview.png",
                self.config.figures_dir / "feature_importance.png",
                self.config.figures_dir / "model_comparison.png",
                self.config.figures_dir / "confusion_matrices.png",
                self.config.figures_dir / "results_summary.png",
                self.config.figures_dir / "cm_lstm.png",
                self.config.figures_dir / "cm_tabnet.png",
            ):
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path), artifact_path="figures")

            run_id = run.info.run_id
            logger.info(f"MLflow run completado: {run_id}")
            return run_id
