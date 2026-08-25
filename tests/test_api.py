import numpy as np
import pytest
from fastapi.testclient import TestClient

from deteccion_fraude.api import app as app_module
from deteccion_fraude.api.schemas import Transaction
from deteccion_fraude.serving import ScoringModel
from tests.test_serving import StubTabNet, build_artifacts, raw_transactions

STUB_METADATA = {
    "name": "fraud_tabnet",
    "version": "3",
    "run_id": "abc123",
    "stage": "Production",
    "flavor": "tabnet",
}


def payload(rows=6):
    frame = raw_transactions(rows=rows, seed=11)
    return {"data": frame.to_dict(orient="records")}


@pytest.fixture
def client(monkeypatch):
    """API con modelo y artefactos simulados, sin tocar el Model Registry."""
    monkeypatch.setattr(
        app_module, "load_model", lambda: ScoringModel(kind="tabnet", model=StubTabNet())
    )
    monkeypatch.setattr(app_module, "load_artifacts", build_artifacts)
    monkeypatch.setattr(app_module, "get_model_metadata", lambda: dict(STUB_METADATA))
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def degraded_client(monkeypatch):
    """API arrancada sin modelo en Production."""
    monkeypatch.setattr(app_module, "load_model", lambda: None)
    monkeypatch.setattr(app_module, "load_artifacts", lambda: None)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_root_reports_served_model(client):
    body = client.get("/").json()
    assert body["status"] == "Online"
    assert body["model_loaded"] is True
    assert body["model_metadata"]["name"] == "fraud_tabnet"
    assert body["model_metadata"]["version"] == "3"


def test_health_is_ok_when_model_and_artifacts_are_loaded(client):
    assert client.get("/health").status_code == 200


def test_health_reports_503_when_degraded(degraded_client):
    response = degraded_client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "Degradado"


def test_predict_returns_one_verdict_per_transaction(client):
    body = client.post("/predict", json=payload(rows=6)).json()
    assert body["total_predictions"] == 6
    assert body["scored_from_index"] == 0
    assert [item["index"] for item in body["results"]] == list(range(6))
    assert body["decision_threshold"] == pytest.approx(0.42)
    assert body["model_metadata"]["run_id"] == "abc123"


def test_predict_applies_the_persisted_threshold(client):
    """StubTabNet devuelve 0.1..0.9 lineal; el umbral 0.42 separa el lote."""
    results = client.post("/predict", json=payload(rows=9)).json()["results"]
    probabilities = [item["fraud_probability"] for item in results]
    codes = [item["prediction_code"] for item in results]
    assert codes == [int(probability > 0.42) for probability in probabilities]
    assert all(item["diagnosis"] == "Fraude (1)" for item in results if item["prediction_code"])


def test_predict_flags_high_risk_transactions(client):
    results = client.post("/predict", json=payload(rows=9)).json()["results"]
    for item in results:
        assert item["high_risk_flag"] == int(item["fraud_probability"] >= 0.85)


def test_predict_confidence_matches_the_winning_class(client):
    results = client.post("/predict", json=payload(rows=5)).json()["results"]
    for item in results:
        probability = item["fraud_probability"]
        expected = probability if item["prediction_code"] else 1 - probability
        assert item["confidence_score"] == pytest.approx(round(expected * 100, 2))


def test_predict_returns_503_without_model(degraded_client):
    response = degraded_client.post("/predict", json=payload())
    assert response.status_code == 503
    assert "fraude promote" in response.json()["detail"]


def test_predict_rejects_empty_batch(client):
    assert client.post("/predict", json={"data": []}).status_code == 422


def test_predict_rejects_negative_amount(client):
    body = payload(rows=2)
    body["data"][0]["Amount"] = -5.0
    assert client.post("/predict", json=body).status_code == 422


def test_predict_rejects_missing_feature(client):
    body = payload(rows=2)
    del body["data"][0]["V14"]
    assert client.post("/predict", json=body).status_code == 422


def test_lstm_response_explains_unscored_prefix(monkeypatch):
    class StubLSTM:
        def predict(self, windows, verbose=0):
            return np.full((len(windows), 1), 0.9)

    monkeypatch.setattr(
        app_module, "load_model", lambda: ScoringModel(kind="lstm", model=StubLSTM())
    )
    monkeypatch.setattr(app_module, "load_artifacts", build_artifacts)
    monkeypatch.setattr(app_module, "get_model_metadata", lambda: dict(STUB_METADATA))

    with TestClient(app_module.app) as client:
        body = client.post("/predict", json=payload(rows=12)).json()

    assert body["scored_from_index"] == 4
    assert body["total_predictions"] == 8
    assert [item["index"] for item in body["results"]] == list(range(4, 12))
    assert "ventanas" in body["message"]


def test_lstm_batch_too_short_returns_422(monkeypatch):
    class StubLSTM:
        def predict(self, windows, verbose=0):
            return np.zeros((len(windows), 1))

    monkeypatch.setattr(
        app_module, "load_model", lambda: ScoringModel(kind="lstm", model=StubLSTM())
    )
    monkeypatch.setattr(app_module, "load_artifacts", build_artifacts)
    monkeypatch.setattr(app_module, "get_model_metadata", lambda: dict(STUB_METADATA))

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload(rows=3))

    assert response.status_code == 422
    assert "al menos" in response.json()["detail"]


def test_transaction_accepts_dataset_aliases_and_field_names():
    dataset_style = raw_transactions(rows=1).to_dict(orient="records")[0]
    by_alias = Transaction(**dataset_style)
    by_field = Transaction(
        **{
            "time": dataset_style["Time"],
            "amount": dataset_style["Amount"],
            **{f"v{i}": dataset_style[f"V{i}"] for i in range(1, 29)},
        }
    )
    assert by_alias == by_field
    assert by_alias.model_dump(by_alias=True)["V14"] == dataset_style["V14"]
