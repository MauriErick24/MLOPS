"""API de inferencia para el modelo de deteccion de fraude en Production."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from loguru import logger
import pandas as pd

from deteccion_fraude.api.model_loader import get_model_metadata, load_artifacts, load_model
from deteccion_fraude.api.schemas import (
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
    ServiceStatus,
    TransactionPrediction,
)
from deteccion_fraude.serving import score_transactions

#: Probabilidad a partir de la cual la transaccion se marca para revision manual.
HIGH_RISK_THRESHOLD = 0.85

#: Literal en vez de `status.HTTP_422_*`, cuyo nombre cambia entre versiones de Starlette.
HTTP_UNPROCESSABLE = 422

_state: dict[str, object] = {"model": None, "artifacts": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carga modelo y preprocesamiento una sola vez, al arrancar el proceso."""
    _state["model"] = load_model()
    _state["artifacts"] = load_artifacts()

    metadata = get_model_metadata()
    if _state["model"] is None:
        logger.error("La API arranco sin modelo: /predict respondera 503.")
    elif _state["artifacts"] is None:
        logger.error("La API arranco sin artefactos de inferencia: /predict respondera 503.")
    else:
        logger.success(f"Modelo {metadata['name']} v{metadata['version']} listo para inferencia.")
    yield
    _state["model"] = None
    _state["artifacts"] = None


app = FastAPI(
    title="API de Deteccion de Fraude",
    description=(
        "Sirve el modelo promovido a Production en el MLflow Model Registry, "
        "reaplicando el preprocesamiento fijado durante el entrenamiento."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _status_payload() -> ServiceStatus:
    model = _state["model"]
    artifacts = _state["artifacts"]
    ready = model is not None and artifacts is not None
    return ServiceStatus(
        status="Online" if ready else "Degradado",
        model_loaded=model is not None,
        artifacts_loaded=artifacts is not None,
        model_metadata=ModelMetadata(**get_model_metadata()),
    )


@app.get("/", response_model=ServiceStatus)
def read_root() -> ServiceStatus:
    """Identidad del modelo servido."""
    return _status_payload()


@app.get("/health", response_model=ServiceStatus)
def health(response: Response) -> ServiceStatus:
    """Readiness probe: 503 mientras falte el modelo o el preprocesamiento."""
    payload = _status_payload()
    if payload.status != "Online":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Puntua un lote cronologico de transacciones."""
    model = _state["model"]
    artifacts = _state["artifacts"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No hay modelo en Production. Ejecute 'fraude train' y "
                "'fraude promote', luego reinicie la API."
            ),
        )
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Faltan los artefactos de inferencia (models/serving_artifacts.pkl). "
                "Ejecute 'fraude train' para regenerarlos."
            ),
        )

    raw = pd.DataFrame([item.model_dump(by_alias=True) for item in payload.data])

    try:
        probabilities, offset = score_transactions(artifacts, model, raw)
    except ValueError as error:
        raise HTTPException(status_code=HTTP_UNPROCESSABLE, detail=str(error)) from error
    except Exception as error:  # cualquier fallo inesperado se traduce a 500
        logger.exception("Fallo la inferencia")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {error}",
        ) from error

    threshold = artifacts.threshold_for(model.kind)
    results = []
    for position, probability in enumerate(probabilities):
        probability = float(probability)
        is_fraud = int(probability > threshold)
        confidence = probability if is_fraud else 1.0 - probability
        results.append(
            TransactionPrediction(
                index=position + offset,
                prediction_code=is_fraud,
                diagnosis="Fraude (1)" if is_fraud else "Legitima (0)",
                fraud_probability=round(probability, 6),
                confidence_score=round(confidence * 100, 2),
                high_risk_flag=int(probability >= HIGH_RISK_THRESHOLD),
            )
        )

    message = "Inferencia completada con exito."
    if offset:
        message = (
            f"Inferencia completada; las primeras {offset} transacciones no se puntuaron "
            f"porque el modelo LSTM requiere ventanas de {artifacts.sequence_length}."
        )

    return PredictionResponse(
        model_metadata=ModelMetadata(**get_model_metadata()),
        decision_threshold=round(threshold, 6),
        total_predictions=len(results),
        scored_from_index=offset,
        results=results,
        message=message,
    )
