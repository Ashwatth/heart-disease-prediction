"""Command-line interface for heart disease prediction."""

import click
from pathlib import Path
from loguru import logger
import sys

from .config import Config
from .data_loader import DataLoader
from .model import HeartDiseasePredictor
from .visualizer import Visualizer


@click.group()
@click.version_option(version="0.2.0")
def main():
    """Heart Disease Prediction using Bayesian Networks.
    
    A comprehensive tool for predicting heart disease using probabilistic graphical models.
    """
    pass


@main.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to the heart disease dataset CSV file",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to configuration file (optional)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory for results and visualizations",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing (default: 0.2)",
)
@click.option(
    "--estimator",
    type=click.Choice(["mle", "bayes"], case_sensitive=False),
    default="mle",
    help="Estimation method for learning CPDs",
)
@click.option(
    "--save-model",
    type=click.Path(),
    default=None,
    help="Path to save the trained model",
)
@click.option("--visualize", is_flag=True, help="Generate visualizations")
def train(data, config, output, test_size, estimator, save_model, visualize):
    """Train a heart disease prediction model."""
    try:
        # Initialize configuration
        cfg = Config(config)
        logger.info("Starting model training...")

        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and preprocess data
        loader = DataLoader(cfg._config)
        logger.info(f"Loading data from {data}")
        raw_data = loader.load_data(data)
        
        if not loader.validate_data(raw_data):
            logger.error("Data validation failed")
            sys.exit(1)

        processed_data = loader.preprocess(raw_data)
        train_data, test_data = loader.split_data(processed_data, test_size=test_size)

        # Build and train model
        model_structure = cfg.get("model.structure")
        predictor = HeartDiseasePredictor(model_structure)
        predictor.fit(train_data, estimator=estimator)

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = predictor.evaluate(test_data)

        # Print results
        click.echo("\n" + "=" * 50)
        click.echo("MODEL EVALUATION RESULTS")
        click.echo("=" * 50)
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                click.echo(f"{metric.capitalize():.<30} {value:.4f}")

        # Save model
        if save_model:
            predictor.save_model(save_model)
            click.echo(f"\nModel saved to: {save_model}")

        # Generate visualizations
        if visualize:
            logger.info("Generating visualizations...")
            viz = Visualizer()
            viz.set_output_dir(output_dir)
            
            viz.plot_data_distribution(processed_data, save=True)
            viz.plot_metrics(metrics, save=True)
            viz.plot_feature_correlation(processed_data, save=True)
            viz.plot_network_structure(model_structure, save=True)
            
            if "confusion_matrix" in metrics:
                import numpy as np
                cm = np.array(metrics["confusion_matrix"])
                viz.plot_confusion_matrix(cm, save=True)

            click.echo(f"\nVisualizations saved to: {output_dir}")

        logger.success("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model file",
)
@click.option("--age", type=int, required=True, help="Patient age")
@click.option("--gender", type=int, required=True, help="Gender (0: female, 1: male)")
@click.option("--cp", type=int, required=True, help="Chest pain type (0-3)")
@click.option("--chol", type=int, help="Serum cholesterol in mg/dl")
@click.option("--restecg", type=int, help="Resting electrocardiographic results (0-2)")
@click.option("--exang", type=int, help="Exercise induced angina (0: no, 1: yes)")
@click.option("--visualize", is_flag=True, help="Show prediction visualization")
def predict(model, age, gender, cp, chol, restecg, exang, visualize):
    """Make a prediction for a single patient."""
    try:
        # Load model
        logger.info(f"Loading model from {model}")
        predictor = HeartDiseasePredictor.load_model(model)

        # Prepare evidence
        evidence = {"age": age, "gender": gender, "cp": cp}
        
        if chol is not None:
            evidence["chol"] = chol
        if restecg is not None:
            evidence["restecg"] = restecg
        if exang is not None:
            evidence["exang"] = exang

        # Make prediction
        logger.info("Making prediction...")
        prediction = predictor.predict(evidence)

        # Display results
        click.echo("\n" + "=" * 50)
        click.echo("HEART DISEASE PREDICTION")
        click.echo("=" * 50)
        click.echo("\nPatient Information:")
        for key, value in evidence.items():
            click.echo(f"  {key:.<20} {value}")

        click.echo("\nPrediction Probabilities:")
        for level, prob in sorted(prediction.items()):
            click.echo(f"  Level {level}:  {prob:.4f} ({prob*100:.2f}%)")

        # Determine most likely outcome
        most_likely = max(prediction.items(), key=lambda x: x[1])
        click.echo(f"\nMost Likely: Level {most_likely[0]} with {most_likely[1]*100:.2f}% probability")

        # Visualize if requested
        if visualize:
            viz = Visualizer()
            viz.plot_prediction_probabilities(prediction, title="Heart Disease Prediction", save=False)

        logger.success("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset CSV file",
)
def info(data):
    """Display information about a dataset."""
    try:
        loader = DataLoader()
        logger.info(f"Loading data from {data}")
        dataset = loader.load_data(data)

        info_dict = loader.get_data_info(dataset)

        click.echo("\n" + "=" * 50)
        click.echo("DATASET INFORMATION")
        click.echo("=" * 50)
        click.echo(f"\nShape: {info_dict['shape'][0]} rows × {info_dict['shape'][1]} columns")
        
        click.echo("\nColumns:")
        for col in info_dict['columns']:
            dtype = info_dict['dtypes'][col]
            missing = info_dict['missing_values'][col]
            click.echo(f"  {col:.<20} {str(dtype):.<15} (Missing: {missing})")

        click.echo("\nData Validation:")
        is_valid = loader.validate_data(dataset)
        click.echo(f"  Valid for training: {'✓ Yes' if is_valid else '✗ No'}")

        logger.success("Info retrieval completed!")

    except Exception as e:
        logger.error(f"Failed to get info: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset CSV file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="visualizations",
    help="Output directory for visualizations",
)
def visualize(data, output):
    """Generate visualizations for a dataset."""
    try:
        # Load data
        loader = DataLoader()
        logger.info(f"Loading data from {data}")
        dataset = loader.load_data(data)
        processed_data = loader.preprocess(dataset)

        # Create visualizations
        viz = Visualizer()
        viz.set_output_dir(output)

        logger.info("Generating visualizations...")
        viz.plot_data_distribution(processed_data, save=True)
        viz.plot_feature_correlation(processed_data, save=True)

        click.echo(f"\nVisualizations saved to: {output}")
        logger.success("Visualization completed!")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
