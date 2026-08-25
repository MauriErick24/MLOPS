"""Contratos de entrada y salida de la API (Pydantic v2).

Los nombres de campo son snake_case y el alias reproduce la columna exacta del
dataset (`V1`, `Amount`, ...). Con `populate_by_name` el cliente puede enviar
cualquiera de las dos formas, y `model_dump(by_alias=True)` reconstruye el
DataFrame con los nombres que espera el feature engineering.
"""

from pydantic import BaseModel, ConfigDict, Field

_V_DESCRIPTION = "Componente PCA anonimizada del dataset original."


class Transaction(BaseModel):
    """Una transaccion cruda de tarjeta de credito."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    time: float = Field(
        ...,
        alias="Time",
        ge=0,
        description="Segundos transcurridos desde la primera transaccion del dataset.",
    )
    v1: float = Field(..., alias="V1", description=_V_DESCRIPTION)
    v2: float = Field(..., alias="V2", description=_V_DESCRIPTION)
    v3: float = Field(..., alias="V3", description=_V_DESCRIPTION)
    v4: float = Field(..., alias="V4", description=_V_DESCRIPTION)
    v5: float = Field(..., alias="V5", description=_V_DESCRIPTION)
    v6: float = Field(..., alias="V6", description=_V_DESCRIPTION)
    v7: float = Field(..., alias="V7", description=_V_DESCRIPTION)
    v8: float = Field(..., alias="V8", description=_V_DESCRIPTION)
    v9: float = Field(..., alias="V9", description=_V_DESCRIPTION)
    v10: float = Field(..., alias="V10", description=_V_DESCRIPTION)
    v11: float = Field(..., alias="V11", description=_V_DESCRIPTION)
    v12: float = Field(..., alias="V12", description=_V_DESCRIPTION)
    v13: float = Field(..., alias="V13", description=_V_DESCRIPTION)
    v14: float = Field(..., alias="V14", description=_V_DESCRIPTION)
    v15: float = Field(..., alias="V15", description=_V_DESCRIPTION)
    v16: float = Field(..., alias="V16", description=_V_DESCRIPTION)
    v17: float = Field(..., alias="V17", description=_V_DESCRIPTION)
    v18: float = Field(..., alias="V18", description=_V_DESCRIPTION)
    v19: float = Field(..., alias="V19", description=_V_DESCRIPTION)
    v20: float = Field(..., alias="V20", description=_V_DESCRIPTION)
    v21: float = Field(..., alias="V21", description=_V_DESCRIPTION)
    v22: float = Field(..., alias="V22", description=_V_DESCRIPTION)
    v23: float = Field(..., alias="V23", description=_V_DESCRIPTION)
    v24: float = Field(..., alias="V24", description=_V_DESCRIPTION)
    v25: float = Field(..., alias="V25", description=_V_DESCRIPTION)
    v26: float = Field(..., alias="V26", description=_V_DESCRIPTION)
    v27: float = Field(..., alias="V27", description=_V_DESCRIPTION)
    v28: float = Field(..., alias="V28", description=_V_DESCRIPTION)
    amount: float = Field(
        ...,
        alias="Amount",
        ge=0,
        description="Monto de la transaccion en la moneda original.",
    )


class PredictionRequest(BaseModel):
    """Lote de transacciones a puntuar.

    El orden importa: las variables de ventana y el modelo LSTM tratan la lista
    como una secuencia cronologica.
    """

    data: list[Transaction] = Field(..., min_length=1)


class ModelMetadata(BaseModel):
    """Identidad del modelo que produjo la inferencia."""

    name: str
    version: str
    run_id: str
    stage: str
    flavor: str


class TransactionPrediction(BaseModel):
    """Veredicto para una transaccion del lote."""

    index: int = Field(..., description="Posicion en la lista enviada.")
    prediction_code: int = Field(..., description="1 = fraude, 0 = legitima.")
    diagnosis: str
    fraud_probability: float = Field(..., description="Probabilidad de la clase fraude.")
    confidence_score: float = Field(..., description="Confianza en el veredicto, en porcentaje.")
    high_risk_flag: int = Field(..., description="1 si la probabilidad supera 0.85.")


class PredictionResponse(BaseModel):
    """Respuesta completa de `/predict`."""

    model_config = ConfigDict(protected_namespaces=())

    model_metadata: ModelMetadata
    decision_threshold: float
    total_predictions: int
    scored_from_index: int = Field(
        ...,
        description=(
            "Primera transaccion con score. El LSTM necesita una ventana previa, "
            "asi que las anteriores quedan sin puntuar."
        ),
    )
    results: list[TransactionPrediction]
    message: str


class ServiceStatus(BaseModel):
    """Respuesta de `/` y `/health`."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    artifacts_loaded: bool
    model_metadata: ModelMetadata
