"""Visualizaciones exploratorias y de evaluacion."""

import matplotlib

matplotlib.use("Agg")

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from deteccion_fraude.config import ExperimentConfig


class FraudVisualizer:
    """Genera graficos de evaluacion del experimento."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @staticmethod
    def summarize(dataframe: pd.DataFrame) -> dict[str, int]:
        total = len(dataframe)
        frauds = int(dataframe["Class"].sum())
        normal = total - frauds
        print(f"Total: {total:,}")
        print(f"Normales: {normal:,} ({normal / total * 100:.2f}%)")
        print(f"Fraudes: {frauds:,} ({frauds / total * 100:.3f}%)")
        print(f"Nulos: {dataframe.isnull().sum().sum()}")
        print(f"Duplicados: {dataframe.duplicated().sum()}")
        return {"total": total, "normal": normal, "frauds": frauds}

    def plot_overview(self, dataframe: pd.DataFrame) -> None:
        stats = self.summarize(dataframe)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ["steelblue", "crimson"]
        axes[0, 0].bar(["Normal", "Fraude"], [stats["normal"], stats["frauds"]], color=colors)
        axes[0, 0].set_title("Distribucion de clases")

        box_data = dataframe.copy()
        box_data["Class"] = box_data["Class"].map({0: "Normal", 1: "Fraude"})
        sns.boxplot(x="Class", y="Amount", data=box_data, ax=axes[0, 1], palette=colors)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Amount por clase (escala log)")

        for class_id, label, color in [
            (0, "Normal", "steelblue"),
            (1, "Fraude", "crimson"),
        ]:
            hours = dataframe[dataframe["Class"] == class_id]["Time"] / 3600
            axes[1, 0].hist(hours, bins=48, alpha=0.6, label=label, color=color, density=True)
        axes[1, 0].legend()
        axes[1, 0].set_title("Distribucion temporal")

        correlations = dataframe.corr(numeric_only=True)["Class"].drop("Class")
        correlations = correlations.sort_values()
        top = pd.concat([correlations.head(10), correlations.tail(10)])
        top.plot.barh(
            ax=axes[1, 1],
            color=["crimson" if value < 0 else "steelblue" for value in top],
        )
        axes[1, 1].set_title("Correlaciones con Class")
        plt.tight_layout()
        path = self.config.figures_dir / "overview.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Overview guardado: {path}")

    @staticmethod
    def plot_feature_importance(selector) -> None:
        _fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        selector.feature_importance.tail(25).plot.barh(ax=axes[0], color="teal")
        axes[0].set_title("Feature importance")
        axes[1].plot(
            range(1, len(selector.cumulative_importance) + 1),
            selector.cumulative_importance,
            "b-o",
            markersize=2,
        )
        axes[1].axhline(95, color="red", linestyle="--")
        axes[1].set_title("Importancia acumulada")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_model_comparison(y_true, scores: dict[str, np.ndarray], results: dict) -> None:
        _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, probabilities in scores.items():
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            precision, recall, _ = precision_recall_curve(y_true, probabilities)
            axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, probabilities):.4f})")
            axes[1].plot(
                recall,
                precision,
                label=f"{name} ({average_precision_score(y_true, probabilities):.4f})",
            )
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set_title("Curva ROC")
        axes[1].set_title("Curva Precision-Recall")
        for axis in axes:
            axis.legend()
            axis.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        _fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
        for axis, (name, result), color in zip(
            np.atleast_1d(axes), results.items(), ["Blues", "Oranges"]
        ):
            sns.heatmap(
                result["cm"],
                annot=True,
                fmt="d",
                cmap=color,
                ax=axis,
                xticklabels=["Normal", "Fraude"],
                yticklabels=["Normal", "Fraude"],
            )
            axis.set_title(
                f"{name}\nP={result['precision']:.3f} "
                f"R={result['recall']:.3f} F1={result['f1']:.3f}"
            )
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Matriz de Confusion — {model_name}")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        path = self.config.figures_dir / f"cm_{model_name.lower()}.png"
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
        path = self.config.figures_dir / "results_summary.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Resumen guardado: {path}")
