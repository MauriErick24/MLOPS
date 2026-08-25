"""Carga del modelo en Production desde el MLflow Model Registry.

Centraliza la resolucion de version y flavor para que la app FastAPI no
dependa de detalles de MLflow.
"""

import os

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.keras
import mlflow.pyfunc
import mlflow.sklearn

from deteccion_fraude.config import MODELS_DIR, PROJ_ROOT, ExperimentConfig
from deteccion_fraude.serving import ScoringModel, ServingArtifacts

#: Modelos que `fraude promote` puede registrar, en orden de preferencia.
CANDIDATE_MODELS: tuple[str, ...] = ("fraud_tabnet", "fraud_lstm")

#: Nombre configurado; "auto" resuelve el ganador presente en el Registry.
MODEL_NAME: str = os.getenv("FRAUD_MODEL_NAME", "auto")

UNKNOWN_METADATA: dict[str, str] = {
    "name": "Desconocido",
    "version": "Desconocida",
    "run_id": "Desconocido",
    "stage": "Desconocido",
    "flavor": "Desconocido",
}

_metadata: dict[str, str] = dict(UNKNOWN_METADATA)


def _tracking_uri() -> str:
    """URI de tracking absoluta, para no depender del directorio de trabajo."""
    override = os.getenv("MLFLOW_TRACKING_URI")
    if override:
        return override
    uri = ExperimentConfig().mlflow_tracking_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith("sqlite:////"):
        return f"{prefix}{(PROJ_ROOT / uri[len(prefix) :]).as_posix()}"
    return uri


def _model_kind(name: str) -> str:
    if "tabnet" in name:
        return "tabnet"
    if "lstm" in name:
        return "lstm"
    return "pyfunc"


def _resolve_version(client, name: str):
    """Devuelve la version en Production o, si no hay, la mas reciente."""
    try:
        production = client.get_latest_versions(name, stages=["Production"])
    except MlflowException:
        return None
    if production:
        return production[0]
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        return None
    return max(versions, key=lambda version: int(version.version)) if versions else None


def _candidates() -> tuple[str, ...]:
    return CANDIDATE_MODELS if MODEL_NAME == "auto" else (MODEL_NAME,)


def load_model() -> ScoringModel | None:
    """Carga el modelo promovido y memoriza sus metadatos.

    Devuelve None si el Registry esta vacio o es inalcanzable; la API arranca
    igual y responde 503 hasta que exista un modelo.
    """
    global _metadata
    _metadata = dict(UNKNOWN_METADATA)

    mlflow.set_tracking_uri(_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    resolved = [
        (name, version)
        for name in _candidates()
        if (version := _resolve_version(client, name)) is not None
    ]
    if not resolved:
        logger.warning(
            f"No hay modelos {list(_candidates())} en el Registry. "
            "Ejecute 'fraude train' y 'fraude promote'."
        )
        return None

    name, version = max(resolved, key=lambda item: item[1].last_updated_timestamp or 0)
    uri = f"models:/{name}/{version.version}"
    kind = _model_kind(name)

    try:
        if kind == "tabnet":
            model = mlflow.sklearn.load_model(uri)
        elif kind == "lstm":
            model = mlflow.keras.load_model(uri)
        else:
            model = mlflow.pyfunc.load_model(uri)
    except Exception as error:  # noqa: BLE001 - ante cualquier flavor, degradar a pyfunc
        logger.warning(f"Fallo la carga nativa de {uri} ({error}); reintentando como pyfunc.")
        try:
            model = mlflow.pyfunc.load_model(uri)
            kind = "pyfunc"
        except Exception as fallback_error:  # noqa: BLE001 - la API debe arrancar igual
            logger.error(f"No se pudo cargar {uri}: {fallback_error}")
            return None

    _metadata = {
        "name": name,
        "version": str(version.version),
        "run_id": version.run_id or "Desconocido",
        "stage": version.current_stage or "None",
        "flavor": kind,
    }
    logger.info(f"Modelo {name} v{version.version} cargado como flavor '{kind}'.")
    return ScoringModel(kind=kind, model=model)


def get_model_metadata() -> dict[str, str]:
    """Metadatos del ultimo `load_model()`."""
    return dict(_metadata)


def load_artifacts() -> ServingArtifacts | None:
    """Carga el preprocesamiento persistido por el entrenamiento."""
    try:
        artifacts = ServingArtifacts.load(MODELS_DIR)
    except Exception as error:  # noqa: BLE001 - la API arranca aunque falten artefactos
        logger.error(f"No se pudieron cargar los artefactos de inferencia: {error}")
        return None
    logger.info(f"Artefactos de inferencia cargados ({len(artifacts.feature_names)} features).")
    return artifacts
