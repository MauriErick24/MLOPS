"""Tests for the `predict` CLI command.

The command reads a pickled results dictionary from `models/results.pkl` and
prints a metric summary. These tests drive it through Typer's CliRunner with an
isolated models directory.
"""

import pickle

from typer.testing import CliRunner

from deteccion_fraude.config import ExperimentConfig
import deteccion_fraude.modeling.predict as predict_mod

runner = CliRunner()


def _use_models_dir(monkeypatch, models_dir):
    """Point the command at an isolated models directory."""
    monkeypatch.setattr(
        predict_mod, "ExperimentConfig", lambda: ExperimentConfig(models_dir=models_dir)
    )


def test_exits_with_error_when_results_are_missing(tmp_path, monkeypatch):
    _use_models_dir(monkeypatch, tmp_path)
    result = runner.invoke(predict_mod.app, [])
    assert result.exit_code == 1


def test_reports_metrics_when_results_are_present(tmp_path, monkeypatch):
    _use_models_dir(monkeypatch, tmp_path)
    results = {
        "lstm": {"threshold": 0.80, "f1": 0.7692, "roi": 192.0},
        "tabnet": {"threshold": 0.79, "f1": 0.7917, "roi": 242.0},
    }
    with (tmp_path / "results.pkl").open("wb") as handle:
        pickle.dump(results, handle)

    result = runner.invoke(predict_mod.app, [])
    assert result.exit_code == 0


def test_handles_a_partial_results_file(tmp_path, monkeypatch):
    _use_models_dir(monkeypatch, tmp_path)
    with (tmp_path / "results.pkl").open("wb") as handle:
        pickle.dump({"tabnet": {"threshold": 0.79, "f1": 0.79, "roi": 242.0}}, handle)

    result = runner.invoke(predict_mod.app, [])
    assert result.exit_code == 0
