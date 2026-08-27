"""Tests for tracking module: wrappers, lineage, promote, and compare."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import deteccion_fraude.tracking as tracking_mod
from deteccion_fraude.config import ExperimentConfig
from deteccion_fraude.tracking import (
    FraudDecisionWrapper,
    LSTMPyFuncWrapper,
    MLflowFraudTrainer,
    get_lineage_metadata,
)


# ---------------------------------------------------------------------------
# get_lineage_metadata
# ---------------------------------------------------------------------------


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_returns_git_commit_when_available(mock_subprocess):
    mock_subprocess.run.side_effect = [
        MagicMock(stdout="abc1234\n", returncode=0),
        MagicMock(stdout="", returncode=0),
    ]
    result = get_lineage_metadata()
    assert result["git_commit"] == "abc1234"
    assert result["dvc_status"] == "synced"


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_returns_standalone_when_git_fails(mock_subprocess):
    mock_subprocess.run.side_effect = [
        MagicMock(stdout="\n", returncode=0),
        MagicMock(stdout="", returncode=0),
    ]
    result = get_lineage_metadata()
    assert result["git_commit"] == "standalone"


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_dvc_synced_when_no_output(mock_subprocess):
    mock_subprocess.run.side_effect = [
        MagicMock(stdout="abc1234\n", returncode=0),
        MagicMock(stdout="", returncode=0),
    ]
    result = get_lineage_metadata()
    assert result["dvc_status"] == "synced"


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_dvc_uncommitted_when_output_present(mock_subprocess):
    mock_subprocess.run.side_effect = [
        MagicMock(stdout="abc1234\n", returncode=0),
        MagicMock(stdout="modified: data/raw/creditcard.csv", returncode=0),
    ]
    result = get_lineage_metadata()
    assert result["dvc_status"] == "uncommitted"


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_dvc_unavailable(mock_subprocess):
    mock_subprocess.run.side_effect = [
        MagicMock(stdout="abc1234\n", returncode=0),
        OSError("dvc not found"),
    ]
    result = get_lineage_metadata()
    assert result["dvc_status"] == "dvc_unavailable"


@patch("deteccion_fraude.tracking.subprocess")
def test_lineage_git_unavailable(mock_subprocess):
    mock_subprocess.run.side_effect = OSError("git not found")
    result = get_lineage_metadata()
    assert result["git_commit"] == "git_unavailable"


# ---------------------------------------------------------------------------
# FraudDecisionWrapper
# ---------------------------------------------------------------------------


class TestFraudDecisionWrapper:
    def _make_wrapper(self, threshold=0.5):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        return FraudDecisionWrapper(mock_model, decision_threshold=threshold)

    def test_predict_returns_dataframe_with_three_columns(self):
        wrapper = self._make_wrapper()
        result = wrapper.predict(None, pd.DataFrame({"a": [1, 2]}))
        assert list(result.columns) == ["probability", "prediction", "high_risk_flag"]
        assert len(result) == 2

    def test_threshold_applies_correctly(self):
        wrapper = self._make_wrapper(threshold=0.5)
        result = wrapper.predict(None, pd.DataFrame({"a": [1, 2]}))
        assert result["prediction"].iloc[0] == 1
        assert result["prediction"].iloc[1] == 0

    def test_high_risk_flag_at_85(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        wrapper = FraudDecisionWrapper(mock_model)
        result = wrapper.predict(None, pd.DataFrame({"a": [1, 2]}))
        assert result["high_risk_flag"].iloc[0] == 1
        assert result["high_risk_flag"].iloc[1] == 0

    def test_probability_matches_class_1(self):
        wrapper = self._make_wrapper()
        result = wrapper.predict(None, pd.DataFrame({"a": [1, 2]}))
        assert result["probability"].iloc[0] == pytest.approx(0.7)
        assert result["probability"].iloc[1] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# LSTMPyFuncWrapper
# ---------------------------------------------------------------------------


class TestLSTMPyFuncWrapper:
    def _make_wrapper(self, seq_len=5):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1])
        return LSTMPyFuncWrapper(mock_model, sequence_length=seq_len)

    def test_predict_reshapes_2d_to_3d(self):
        wrapper = self._make_wrapper(seq_len=5)
        df = pd.DataFrame(np.ones((10, 3)))
        wrapper.predict(None, df)
        call_args = wrapper.model.predict.call_args[0][0]
        assert call_args.shape == (2, 5, 3)

    def test_predict_with_3d_input(self):
        wrapper = self._make_wrapper(seq_len=3)
        arr_3d = np.ones((2, 3, 4))
        wrapper.model.predict.return_value = np.array([0.5, 0.6])
        wrapper.predict(None, pd.DataFrame(arr_3d.reshape(6, 4)))
        assert wrapper.model.predict.called

    def test_predict_returns_1d_output(self):
        wrapper = self._make_wrapper(seq_len=5)
        df = pd.DataFrame(np.ones((10, 3)))
        result = wrapper.predict(None, df)
        assert result.ndim == 1


# ---------------------------------------------------------------------------
# MLflowFraudTrainer — promote_best_model
# ---------------------------------------------------------------------------


@patch("deteccion_fraude.tracking.mlflow")
@patch("deteccion_fraude.tracking.get_lineage_metadata")
def test_promote_best_model_selects_lstm_when_higher(mock_lineage, mock_mlflow):
    mock_lineage.return_value = {"git_commit": "abc"}
    config = ExperimentConfig()
    trainer = MLflowFraudTrainer(config)

    client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client
    run = MagicMock()
    run.data.metrics = {"lstm_f1": 0.9, "tabnet_f1": 0.7}
    client.get_run.return_value = run

    winner = trainer.promote_best_model("run_123")
    assert winner == "lstm"


@patch("deteccion_fraude.tracking.mlflow")
@patch("deteccion_fraude.tracking.get_lineage_metadata")
def test_promote_best_model_selects_tabnet_when_higher(mock_lineage, mock_mlflow):
    mock_lineage.return_value = {"git_commit": "abc"}
    config = ExperimentConfig()
    trainer = MLflowFraudTrainer(config)

    client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client
    run = MagicMock()
    run.data.metrics = {"lstm_f1": 0.5, "tabnet_f1": 0.8}
    client.get_run.return_value = run

    winner = trainer.promote_best_model("run_456")
    assert winner == "tabnet"


@patch("deteccion_fraude.tracking.mlflow")
@patch("deteccion_fraude.tracking.get_lineage_metadata")
def test_promote_best_model_creates_registered_model_on_first_run(mock_lineage, mock_mlflow):
    import mlflow.exceptions as mlflow_exc

    mock_lineage.return_value = {"git_commit": "abc"}
    config = ExperimentConfig()
    trainer = MLflowFraudTrainer(config)

    client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client
    mock_mlflow.exceptions = mlflow_exc
    run = MagicMock()
    run.data.metrics = {"lstm_f1": 0.9, "tabnet_f1": 0.7}
    client.get_run.return_value = run
    client.get_registered_model.side_effect = mlflow_exc.MlflowException("not found")

    winner = trainer.promote_best_model("run_789")
    client.create_registered_model.assert_called_once()


# ---------------------------------------------------------------------------
# MLflowFraudTrainer — compare_runs
# ---------------------------------------------------------------------------


@patch("deteccion_fraude.tracking.mlflow")
@patch("deteccion_fraude.tracking.get_lineage_metadata")
def test_compare_runs_returns_empty_for_missing_experiment(mock_lineage, mock_mlflow):
    mock_lineage.return_value = {"git_commit": "abc"}
    config = ExperimentConfig()
    trainer = MLflowFraudTrainer(config)

    client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client
    client.get_experiment_by_name.return_value = None

    result = trainer.compare_runs("nonexistent")
    assert result.empty


@patch("deteccion_fraude.tracking.mlflow")
@patch("deteccion_fraude.tracking.get_lineage_metadata")
def test_compare_runs_returns_empty_when_no_runs(mock_lineage, mock_mlflow):
    mock_lineage.return_value = {"git_commit": "abc"}
    config = ExperimentConfig()
    trainer = MLflowFraudTrainer(config)

    client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = client
    exp = MagicMock()
    exp.experiment_id = "1"
    client.get_experiment_by_name.return_value = exp
    client.search_runs.return_value = []

    result = trainer.compare_runs()
    assert result.empty
