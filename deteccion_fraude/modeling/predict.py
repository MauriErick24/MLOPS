from pathlib import Path
import pickle

from loguru import logger
import typer

from deteccion_fraude.config import MODELS_DIR, ExperimentConfig

app = typer.Typer()


@app.command()
def main(
    data_path: Path = MODELS_DIR / "results.pkl",
):
    config = ExperimentConfig()

    results_path = config.models_dir / "results.pkl"
    if not results_path.exists():
        logger.error("No se encontro results.pkl. Ejecute train primero.")
        raise typer.Exit(1)

    with results_path.open("rb") as f:
        results = pickle.load(f)

    for model_name in ("lstm", "tabnet"):
        if model_name in results:
            r = results[model_name]
            logger.info(
                f"{model_name.upper()}: threshold={r['threshold']:.3f}, "
                f"f1={r['f1']:.4f}, roi={r['roi']:.2f}%"
            )
    logger.success("Resumen de modelos cargado.")


if __name__ == "__main__":
    app()
