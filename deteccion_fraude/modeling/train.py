from loguru import logger
import mlflow
import typer

from deteccion_fraude.config import ExperimentConfig

app = typer.Typer()


@app.command()
def main():
    from deteccion_fraude.dataset import FraudDataset
    from deteccion_fraude.features import FraudPreprocessor
    from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline
    from deteccion_fraude.plots import FraudVisualizer

    config = ExperimentConfig()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    df = FraudDataset.load(config.data_file)
    splits = FraudDataset.split(df, config)
    preprocessor = FraudPreprocessor(config)
    data = preprocessor.fit_transform(splits)

    pipeline = FraudDetectionPipeline(config, data)
    pipeline.select_features()
    pipeline.train()
    results = pipeline.evaluate()

    for name, result in results.items():
        logger.info(f"{name}: F1={result['f1']:.4f}, ROI={result['roi']:.2f}%")

    visualizer = FraudVisualizer(config)
    visualizer.plot_overview(df)
    visualizer.plot_feature_importance(pipeline.selector, config.figures_dir)
    visualizer.plot_model_comparison(
        pipeline.y_test_aligned, pipeline.test_scores, results, config.figures_dir
    )
    for name, result in results.items():
        visualizer.plot_confusion_matrix(result["cm"], name)
    visualizer.plot_results_summary(results)

    run_id = pipeline.log_to_mlflow()
    logger.success(f"Entrenamiento completo. MLflow run: {run_id}")


if __name__ == "__main__":
    app()
