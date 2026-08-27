"""Contract tests for the Pydantic v2 API schemas.

These cover the response models (ModelMetadata, TransactionPrediction,
PredictionResponse, ServiceStatus) and the request-level validation rules that
the API relies on.
"""

from pydantic import ValidationError
import pytest

from deteccion_fraude.api.schemas import (
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
    ServiceStatus,
    Transaction,
    TransactionPrediction,
)


def _raw_transaction(**overrides):
    row = {"Time": 0.0, "Amount": 149.62}
    for i in range(1, 29):
        row[f"V{i}"] = 0.1
    row.update(overrides)
    return row


def _metadata():
    return ModelMetadata(
        name="fraud_tabnet", version="1", run_id="abc123", stage="Production", flavor="tabnet"
    )


def _prediction():
    return TransactionPrediction(
        index=0,
        prediction_code=0,
        diagnosis="Legitima (0)",
        fraud_probability=0.0142,
        confidence_score=98.58,
        high_risk_flag=0,
    )


def test_model_metadata_requires_every_field():
    with pytest.raises(ValidationError):
        ModelMetadata(name="fraud_tabnet", version="1")


def test_prediction_response_round_trips_through_a_dict():
    response = PredictionResponse(
        model_metadata=_metadata(),
        decision_threshold=0.284,
        total_predictions=1,
        scored_from_index=0,
        results=[_prediction()],
        message="ok",
    )
    dumped = response.model_dump()
    assert dumped["model_metadata"]["name"] == "fraud_tabnet"
    assert dumped["decision_threshold"] == 0.284
    assert len(dumped["results"]) == 1

    rebuilt = PredictionResponse(**dumped)
    assert rebuilt == response


def test_prediction_response_requires_the_scored_offset():
    with pytest.raises(ValidationError):
        PredictionResponse(
            model_metadata=_metadata(),
            decision_threshold=0.5,
            total_predictions=0,
            results=[],
            message="ok",
        )


def test_service_status_carries_the_model_identity():
    status = ServiceStatus(
        status="Online",
        model_loaded=True,
        artifacts_loaded=True,
        model_metadata=_metadata(),
    )
    assert status.status == "Online"
    assert status.model_metadata.flavor == "tabnet"


def test_prediction_request_rejects_an_empty_batch():
    with pytest.raises(ValidationError):
        PredictionRequest(data=[])


def test_prediction_request_accepts_a_valid_batch():
    request = PredictionRequest(data=[_raw_transaction()])
    assert len(request.data) == 1
    assert isinstance(request.data[0], Transaction)


def test_transaction_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        Transaction(**_raw_transaction(unexpected=1.0))


def test_transaction_rejects_negative_amount_and_time():
    with pytest.raises(ValidationError):
        Transaction(**_raw_transaction(Amount=-1.0))
    with pytest.raises(ValidationError):
        Transaction(**_raw_transaction(Time=-1.0))


def test_transaction_requires_all_pca_components():
    row = _raw_transaction()
    del row["V14"]
    with pytest.raises(ValidationError):
        Transaction(**row)


def test_transaction_accepts_zero_for_time_and_amount():
    transaction = Transaction(**_raw_transaction(Time=0, Amount=0))

    assert transaction.time == 0
    assert transaction.amount == 0


def test_transaction_serializes_using_dataset_aliases():
    transaction = Transaction(**_raw_transaction())

    serialized = transaction.model_dump(by_alias=True)

    assert serialized["Time"] == 0.0
    assert serialized["Amount"] == 149.62
    assert serialized["V28"] == 0.1


def test_prediction_response_preserves_lstm_scored_offset():
    response = PredictionResponse(
        model_metadata=_metadata(),
        decision_threshold=0.5,
        total_predictions=2,
        scored_from_index=4,
        results=[_prediction()],
        message="ok",
    )

    assert response.scored_from_index == 4
