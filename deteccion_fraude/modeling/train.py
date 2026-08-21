"""CLI profesional para deteccion de fraude con MLflow."""

from loguru import logger
import mlflow
import typer

app = typer.Typer(help="Fraud Detection MLOps CLI — train, compare, promote.")


@app.command()
def train(
    experiment: str = typer.Option("deteccion_fraude", help="Nombre del experimento MLflow"),
    run_name: str = typer.Option("fraud_detection_pipeline", help="Nombre del run"),
):
    """Entrena ambos modelos (LSTM + TabNet) y registra en MLflow."""
    from deteccion_fraude.config import ExperimentConfig
    from deteccion_fraude.dataset import FraudDataset
    from deteccion_fraude.features import FraudPreprocessor
    from deteccion_fraude.modeling.pipeline import FraudDetectionPipeline
    from deteccion_fraude.plots import FraudVisualizer

    config = ExperimentConfig(mlflow_experiment_name=experiment)
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

    run_id = pipeline.log_to_mlflow(run_name=run_name)
    logger.success(f"Entrenamiento completo. MLflow run: {run_id}")


@app.command()
def promote(
    experiment: str = typer.Option("deteccion_fraude", help="Nombre del experimento MLflow"),
    run_id: str = typer.Option(None, help="Run ID especifico (default: ultimo run)"),
    metric: str = typer.Option("f1", help="Metrica para seleccionar ganador"),
):
    """Promueve el mejor modelo a Production en MLflow Model Registry."""
    from deteccion_fraude.config import ExperimentConfig
    from deteccion_fraude.tracking import MLflowFraudTrainer

    config = ExperimentConfig(mlflow_experiment_name=experiment)
    trainer = MLflowFraudTrainer(config)

    if run_id is None:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment])
        if not runs:
            logger.error("No hay runs en este experimento")
            raise typer.Exit(code=1)
        run_id = runs[0].info.run_id
        logger.info(f"Usando ultimo run: {run_id}")

    winner = trainer.promote_best_model(run_id, metric=metric)
    logger.success(f"Modelo {winner} promovido a Production")


@app.command()
def compare(
    experiment: str = typer.Option("deteccion_fraude", help="Nombre del experimento MLflow"),
):
    """Compara metricas entre todos los runs del experimento."""
    from deteccion_fraude.config import ExperimentConfig
    from deteccion_fraude.tracking import MLflowFraudTrainer

    config = ExperimentConfig(mlflow_experiment_name=experiment)
    trainer = MLflowFraudTrainer(config)
    trainer.compare_runs()


if __name__ == "__main__":
    app()
