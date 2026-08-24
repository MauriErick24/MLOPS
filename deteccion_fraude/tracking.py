"""MLflow tracking, lineage y serving para deteccion de fraude."""

import subprocess

from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.pyfunc import PythonModel
import mlflow.sklearn
import numpy as np
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

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        probabilities = self.model.predict_proba(model_input)[:, 1]
        decisions = (probabilities >= self.decision_threshold).astype(int)
        return pd.DataFrame(
            {
                "probability": probabilities,
                "prediction": decisions,
                "high_risk_flag": (probabilities >= 0.85).astype(int),
            }
        )


class LSTMPyFuncWrapper(PythonModel):
    """Wrapper PyFunc para LSTM que acepta 2D y convierte a 3D."""

    def __init__(self, trained_model, sequence_length: int = 5):
        self.model = trained_model
        self.sequence_length = sequence_length

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> np.ndarray:
        arr = model_input.values if isinstance(model_input, pd.DataFrame) else model_input
        if len(arr.shape) == 2:
            n_samples = arr.shape[0] // self.sequence_length
            arr = arr[: n_samples * self.sequence_length]
            arr = arr.reshape(n_samples, self.sequence_length, -1)
        return self.model.predict(arr).ravel()


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
        run_name: str = "fraud_detection_pipeline",
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

        with mlflow.start_run(run_name=run_name) as run:
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

    def evaluate_with_mlflow(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> dict:
        """Evaluacion automatizada usando mlflow.evaluate()."""
        wrapped_model = FraudDecisionWrapper(model)
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            eval_result = mlflow.evaluate(
                wrapped_model.predict,
                pd.DataFrame(X_test),
                targets=y_test,
                model_type="classifier",
                evaluators=["default"],
                evaluator_config={"log_model_explainability": False},
            )
            logger.info(f"Evaluacion {model_name}: {eval_result.metrics}")
            return eval_result.metrics

    def promote_best_model(self, run_id: str, metric: str = "f1") -> str:
        """Selecciona el mejor modelo y lo promueve a Production."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        lstm_metric = run.data.metrics.get(f"lstm_{metric}", 0)
        tabnet_metric = run.data.metrics.get(f"tabnet_{metric}", 0)
        winner = "lstm" if lstm_metric >= tabnet_metric else "tabnet"

        model_uri = f"runs:/{run_id}/{winner}_model"
        model_name = f"fraud_{winner}"

        result = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production",
        )
        logger.info(
            f"Modelo ganador: {winner} ({metric}={max(lstm_metric, tabnet_metric):.4f}) "
            f"— version {result.version} en Production"
        )
        return winner

    def compare_runs(self, experiment: str | None = None) -> pd.DataFrame:
        """Compara metricas entre runs del experimento."""
        exp_id = experiment or self.config.mlflow_experiment_name
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[exp_id])
        if not runs:
            logger.warning("No hay runs para comparar")
            return pd.DataFrame()

        rows = []
        for run in runs:
            row = {"run_id": run.info.run_id, "run_name": run.info.run_name}
            row.update(run.data.params)
            row.update(run.data.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        if "run_id" in df.columns:
            df = df.sort_values("f1", ascending=False, na_position="last")
        logger.info(f"Comparacion de {len(df)} runs:")
        print(df.to_string(index=False))
        return df
