from deteccion_fraude.config import ExperimentConfig


def test_feature_columns_are_stable_and_unique():
    config = ExperimentConfig()

    assert len(config.feature_columns) == 50
    assert len(set(config.feature_columns)) == 50


def test_create_output_directories_creates_expected_folders(tmp_path):
    config = ExperimentConfig(
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "reports" / "figures",
    )

    config.create_output_directories()

    assert config.models_dir.exists()
    assert config.figures_dir.exists()