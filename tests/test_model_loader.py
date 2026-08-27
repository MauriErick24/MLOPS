import importlib
import sys
from types import ModuleType



def _import_model_loader():
    previous_modules = {
        name: sys.modules.get(name)
        for name in (
            "mlflow",
            "mlflow.exceptions",
            "mlflow.keras",
            "mlflow.pyfunc",
            "mlflow.sklearn",
            "mlflow.tracking",
            "deteccion_fraude.modeling.lstm",
        )
    }

    stub_mlflow = ModuleType("mlflow")
    stub_mlflow.__path__ = []
    stub_mlflow.set_tracking_uri = lambda uri: None

    stub_exceptions = ModuleType("mlflow.exceptions")

    class MlflowException(Exception):
        pass

    stub_exceptions.MlflowException = MlflowException

    for module_name in ("mlflow.keras", "mlflow.pyfunc", "mlflow.sklearn"):
        submodule = ModuleType(module_name)
        submodule.load_model = lambda *args, **kwargs: object()
        sys.modules[module_name] = submodule

    stub_tracking = ModuleType("mlflow.tracking")
    stub_tracking.MlflowClient = object

    stub_lstm = ModuleType("deteccion_fraude.modeling.lstm")
    stub_lstm.FocalLoss = object
    stub_lstm.focal_loss = lambda *args, **kwargs: object()

    sys.modules["mlflow"] = stub_mlflow
    sys.modules["mlflow.exceptions"] = stub_exceptions
    sys.modules["mlflow.tracking"] = stub_tracking
    sys.modules["deteccion_fraude.modeling.lstm"] = stub_lstm
    stub_mlflow.tracking = stub_tracking
    stub_mlflow.exceptions = stub_exceptions

    try:
        return importlib.import_module("deteccion_fraude.api.model_loader")
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


model_loader = _import_model_loader()


def test_model_kind_maps_known_flavors():
    assert model_loader._model_kind("fraud_tabnet") == "tabnet"
    assert model_loader._model_kind("fraud_lstm") == "lstm"
    assert model_loader._model_kind("fraud_pyfunc") == "pyfunc"


class _Version:
    def __init__(self, version, stage="Production", run_id="run-1", updated=0):
        self.version = version
        self.current_stage = stage
        self.run_id = run_id
        self.last_updated_timestamp = updated


class _Client:
    def __init__(self, production=None, versions=None):
        self.production = production or []
        self.versions = versions or []

    def get_latest_versions(self, name, stages):
        return self.production

    def search_model_versions(self, query):
        return self.versions


def test_resolve_version_prefers_production_over_registry_history():
    production = _Version("2", updated=10)
    older = _Version("1", stage="Archived", updated=1)
    newer = _Version("9", stage="Archived", updated=20)
    client = _Client(production=[production], versions=[older, newer])

    resolved = model_loader._resolve_version(client, "fraud_tabnet")

    assert resolved is production


def test_resolve_version_falls_back_to_latest_registry_entry():
    older = _Version("3", stage="Archived", updated=1)
    newer = _Version("11", stage="Archived", updated=20)
    client = _Client(production=[], versions=[older, newer])

    resolved = model_loader._resolve_version(client, "fraud_tabnet")

    assert resolved is newer


def test_load_artifacts_falls_back_to_disk_when_mlflow_artifact_missing(monkeypatch):
    class _Artifacts:
        feature_names = ["a", "b"]

    class _ClientWithMissingArtifact:
        def download_artifacts(self, run_id, path, destination_dir):
            raise RuntimeError("missing artifact")

    def fake_load(path):
        assert path == model_loader.MODELS_DIR
        return _Artifacts()

    monkeypatch.setattr(model_loader, "_metadata", {"run_id": "run-123"})
    monkeypatch.setattr(
        model_loader.mlflow.tracking,
        "MlflowClient",
        lambda: _ClientWithMissingArtifact(),
    )
    monkeypatch.setattr(model_loader.ServingArtifacts, "load", fake_load)

    artifacts = model_loader.load_artifacts()

    assert isinstance(artifacts, _Artifacts)