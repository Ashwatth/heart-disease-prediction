"""Visualization utilities for the heart disease prediction system."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


class Visualizer:
    """Handle visualizations for the heart disease prediction system."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        sns.set_palette("husl")
        self.figure_dir: Optional[Path] = None

    def set_output_dir(self, directory: str) -> None:
        """Set the output directory for saving figures.

        Args:
            directory: Path to output directory
        """
        self.figure_dir = Path(directory)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to {self.figure_dir}")

    def plot_data_distribution(
        self, data: pd.DataFrame, target: str = "heartdisease", save: bool = False
    ) -> None:
        """Plot distribution of target variable and features.

        Args:
            data: DataFrame to visualize
            target: Target variable name
            save: Whether to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Heart Disease Dataset Distribution", fontsize=16, fontweight="bold")

        # Target distribution
        target_counts = data[target].value_counts().sort_index()
        axes[0, 0].bar(target_counts.index.astype(str), target_counts.values, color="skyblue")
        axes[0, 0].set_title("Heart Disease Distribution")
        axes[0, 0].set_xlabel("Heart Disease Level")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(axis="y", alpha=0.3)

        # Age distribution
        if "age" in data.columns:
            axes[0, 1].hist(data["age"].dropna(), bins=20, color="lightcoral", edgecolor="black")
            axes[0, 1].set_title("Age Distribution")
            axes[0, 1].set_xlabel("Age")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].grid(axis="y", alpha=0.3)

        # Gender distribution
        if "gender" in data.columns:
            gender_counts = data["gender"].value_counts()
            axes[1, 0].pie(
                gender_counts.values,
                labels=[f"Gender {int(x)}" for x in gender_counts.index],
                autopct="%1.1f%%",
                startangle=90,
                colors=["lightblue", "lightpink"],
            )
            axes[1, 0].set_title("Gender Distribution")

        # Chest pain type distribution
        if "cp" in data.columns:
            cp_counts = data["cp"].value_counts().sort_index()
            axes[1, 1].bar(cp_counts.index.astype(str), cp_counts.values, color="lightgreen")
            axes[1, 1].set_title("Chest Pain Type Distribution")
            axes[1, 1].set_xlabel("Chest Pain Type")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save and self.figure_dir:
            filepath = self.figure_dir / "data_distribution.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved data distribution plot to {filepath}")

        plt.show()

    def plot_confusion_matrix(
        self, confusion_matrix: np.ndarray, labels: List[str] = None, save: bool = False
    ) -> None:
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            save: Whether to save the figure
        """
        if labels is None:
            labels = [str(i) for i in range(len(confusion_matrix))]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        if save and self.figure_dir:
            filepath = self.figure_dir / "confusion_matrix.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confusion matrix plot to {filepath}")

        plt.show()

    def plot_metrics(self, metrics: Dict[str, float], save: bool = False) -> None:
        """Plot evaluation metrics.

        Args:
            metrics: Dictionary of metric names and values
            save: Whether to save the figure
        """
        # Filter out non-scalar metrics
        scalar_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))
        }

        if not scalar_metrics:
            logger.warning("No scalar metrics to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(scalar_metrics.keys())
        metric_values = list(scalar_metrics.values())

        bars = ax.bar(metric_names, metric_values, color=sns.color_palette("husl", len(metric_names)))
        ax.set_title("Model Performance Metrics", fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save and self.figure_dir:
            filepath = self.figure_dir / "metrics.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {filepath}")

        plt.show()

    def plot_feature_correlation(self, data: pd.DataFrame, save: bool = False) -> None:
        """Plot correlation heatmap of features.

        Args:
            data: DataFrame with features
            save: Whether to save the figure
        """
        # Select numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation plot")
            return

        correlation_matrix = data[numeric_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()

        if save and self.figure_dir:
            filepath = self.figure_dir / "correlation_heatmap.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved correlation heatmap to {filepath}")

        plt.show()

    def plot_network_structure(
        self, model_structure: List[tuple], save: bool = False
    ) -> None:
        """Plot Bayesian Network structure.

        Args:
            model_structure: List of edges (parent, child)
            save: Whether to save the figure
        """
        try:
            import networkx as nx

            G = nx.DiGraph()
            G.add_edges_from(model_structure)

            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=2, iterations=50)

            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            nx.draw_networkx_edges(
                G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->"
            )

            plt.title("Bayesian Network Structure", fontsize=16, fontweight="bold", pad=20)
            plt.axis("off")
            plt.tight_layout()

            if save and self.figure_dir:
                filepath = self.figure_dir / "network_structure.png"
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Saved network structure plot to {filepath}")

            plt.show()

        except ImportError:
            logger.warning("NetworkX not installed. Cannot plot network structure.")
            logger.info("Install with: pip install networkx")

    def plot_prediction_probabilities(
        self, predictions: Dict[str, float], title: str = "Prediction Probabilities", save: bool = False
    ) -> None:
        """Plot prediction probabilities.

        Args:
            predictions: Dictionary of class labels and probabilities
            title: Plot title
            save: Whether to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = list(predictions.keys())
        probabilities = list(predictions.values())

        bars = ax.bar(labels, probabilities, color=sns.color_palette("viridis", len(labels)))
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Heart Disease Level", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        if save and self.figure_dir:
            filepath = self.figure_dir / "prediction_probabilities.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved prediction probabilities plot to {filepath}")

        plt.show()
