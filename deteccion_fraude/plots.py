from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pickle

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from deteccion_fraude.config import FIGURES_DIR, ExperimentConfig

app = typer.Typer()


class FraudVisualizer:
    """Genera graficos de evaluacion del experimento."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Matriz de Confusion — {model_name}")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        path = (
            self.config.models_dir.parent / "reports" / "figures" / f"cm_{model_name.lower()}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Matriz de confusion guardada: {path}")

    def plot_results_summary(self, results: dict) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        models = list(results.keys())
        metrics = ["f1", "precision", "recall"]
        for i, metric in enumerate(metrics):
            vals = [results[m][metric] for m in models]
            axes[0].bar([f"{m}\n{metric}" for m in models], vals)
        axes[0].set_title("Metricas por Modelo")
        axes[0].set_ylim(0, 1.05)

        rois = [results[m]["roi"] for m in models]
        axes[1].bar(models, rois, color=["steelblue", "coral"])
        axes[1].set_title("ROI (%) por Modelo")
        axes[1].set_ylabel("ROI %")
        path = self.config.models_dir.parent / "reports" / "figures" / "results_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Resumen guardado: {path}")


@app.command()
def main(
    results_path: Path = FIGURES_DIR.parent.parent / "models" / "results.pkl",
):
    config = ExperimentConfig()

    if not results_path.exists():
        logger.error("No se encontro results.pkl. Ejecute train primero.")
        raise typer.Exit(1)

    with results_path.open("rb") as f:
        results = pickle.load(f)

    viz = FraudVisualizer(config)
    for model_name in ("lstm", "tabnet"):
        if model_name in results and "cm" in results[model_name]:
            viz.plot_confusion_matrix(results[model_name]["cm"], model_name.upper())
    viz.plot_results_summary(results)
    logger.success("Visualizaciones generadas.")


if __name__ == "__main__":
    app()
