from pathlib import Path

from loguru import logger
import typer

from deteccion_fraude.config import RAW_DATA_DIR, ExperimentConfig

app = typer.Typer()


@app.command()
def main(
    data_path: Path = RAW_DATA_DIR / "dataset.csv",
):
    from deteccion_fraude.dataset import FraudDataset
    from deteccion_fraude.features import FraudPreprocessor
    from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline

    config = ExperimentConfig()
    df = FraudDataset.load(data_path)
    splits = FraudDataset.split(df, config)
    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform(splits)

    pipeline = FraudDetectionPipeline(config, data)
    pipeline.select_features()
    pipeline.train()
    results = pipeline.evaluate()
    pipeline.save_artifacts()

    for name, result in results.items():
        logger.info(f"{name}: F1={result['f1']:.4f}, ROI={result['roi']:.2f}%")
    logger.success("Entrenamiento completo.")


if __name__ == "__main__":
    app()
