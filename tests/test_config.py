"""Tests for ExperimentConfig properties and directory creation."""

from pathlib import Path

from deteccion_fraude.config import ExperimentConfig


def test_feature_columns_has_50_elements():
    config = ExperimentConfig()
    assert len(config.feature_columns) == 50


def test_feature_columns_are_unique():
    config = ExperimentConfig()
    assert len(config.feature_columns) == len(set(config.feature_columns))


def test_pca_columns_are_v1_to_v28():
    config = ExperimentConfig()
    assert config.pca_columns == [f"V{i}" for i in range(1, 29)]


def test_robust_columns_are_15():
    config = ExperimentConfig()
    assert len(config.robust_columns) == 15


def test_standard_columns_are_7():
    config = ExperimentConfig()
    assert len(config.standard_columns) == 7


def test_feature_columns_union_of_three_groups():
    config = ExperimentConfig()
    expected = config.pca_columns + config.robust_columns + config.standard_columns
    assert config.feature_columns == expected


def test_create_output_directories(tmp_path):
    config = ExperimentConfig(models_dir=tmp_path / "models", figures_dir=tmp_path / "figures")
    config.create_output_directories()
    assert config.models_dir.is_dir()
    assert config.figures_dir.is_dir()


def test_create_output_directories_idempotent(tmp_path):
    config = ExperimentConfig(models_dir=tmp_path / "models", figures_dir=tmp_path / "figures")
    config.create_output_directories()
    config.create_output_directories()
    assert config.models_dir.is_dir()
