"""Heart Disease Prediction Model using Bayesian Networks."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from loguru import logger


class HeartDiseasePredictor:
    """Bayesian Network model for heart disease prediction."""

    def __init__(self, model_structure: List[Tuple[str, str]] = None):
        """Initialize the predictor.

        Args:
            model_structure: List of tuples defining the network structure
                            Each tuple is (parent, child) edge
        """
        if model_structure is None:
            # Default structure from original code
            model_structure = [
                ("age", "heartdisease"),
                ("gender", "heartdisease"),
                ("exang", "heartdisease"),
                ("cp", "heartdisease"),
                ("heartdisease", "restecg"),
                ("heartdisease", "chol"),
            ]

        self.model_structure = model_structure
        self.model: Optional[BayesianModel] = None
        self.inference_engine: Optional[VariableElimination] = None
        self._is_fitted = False

    def build_model(self) -> BayesianModel:
        """Build the Bayesian Network model structure.

        Returns:
            The Bayesian Network model
        """
        self.model = BayesianModel(self.model_structure)
        logger.info(f"Built Bayesian Network with {len(self.model_structure)} edges")
        return self.model

    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "mle",
        prior_type: str = "BDeu",
        equivalent_sample_size: int = 10,
    ) -> None:
        """Fit the model using training data.

        Args:
            data: Training data DataFrame
            estimator: Estimator type ('mle' for Maximum Likelihood or 'bayes' for Bayesian)
            prior_type: Prior type for Bayesian estimation ('BDeu', 'K2', 'dirichlet')
            equivalent_sample_size: Equivalent sample size for BDeu prior
        """
        if self.model is None:
            self.build_model()

        try:
            if estimator.lower() == "mle":
                logger.info("Fitting model using Maximum Likelihood Estimation")
                self.model.fit(data, estimator=MaximumLikelihoodEstimator)
            elif estimator.lower() == "bayes":
                logger.info(f"Fitting model using Bayesian Estimation with {prior_type} prior")
                self.model.fit(
                    data,
                    estimator=BayesianEstimator,
                    prior_type=prior_type,
                    equivalent_sample_size=equivalent_sample_size,
                )
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            self.inference_engine = VariableElimination(self.model)
            self._is_fitted = True
            logger.success("Model fitted successfully")

        except Exception as e:
            logger.error(f"Failed to fit model: {e}")
            raise

    def predict(self, evidence: Dict[str, Any], target: str = "heartdisease") -> Dict[str, float]:
        """Make a prediction given evidence.

        Args:
            evidence: Dictionary of evidence variables and their values
            target: Target variable to predict

        Returns:
            Dictionary with prediction probabilities for each state
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        try:
            result = self.inference_engine.query(variables=[target], evidence=evidence)
            
            # Convert to dictionary
            prediction = {}
            for i, value in enumerate(result.values):
                prediction[str(result.state_names[target][i])] = float(value)

            logger.debug(f"Prediction for evidence {evidence}: {prediction}")
            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(
        self, data: pd.DataFrame, target: str = "heartdisease"
    ) -> List[Dict[str, float]]:
        """Make predictions for multiple samples.

        Args:
            data: DataFrame with evidence variables
            target: Target variable to predict

        Returns:
            List of prediction dictionaries
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        predictions = []
        evidence_cols = [col for col in data.columns if col != target]

        for idx, row in data.iterrows():
            evidence = {col: row[col] for col in evidence_cols if not pd.isna(row[col])}
            try:
                pred = self.predict(evidence, target)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict for row {idx}: {e}")
                predictions.append({})

        return predictions

    def evaluate(self, test_data: pd.DataFrame, target: str = "heartdisease") -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            test_data: Test dataset
            target: Target variable name

        Returns:
            Dictionary with evaluation metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        logger.info("Evaluating model performance...")

        # Get predictions
        predictions = self.predict_batch(test_data, target)
        
        # Extract predicted classes (class with highest probability)
        y_pred = []
        for pred in predictions:
            if pred:
                y_pred.append(max(pred.items(), key=lambda x: x[1])[0])
            else:
                y_pred.append(None)

        # Get actual values
        y_true = test_data[target].values

        # Filter out None predictions
        valid_indices = [i for i, p in enumerate(y_pred) if p is not None]
        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]

        # Convert to numeric if needed
        try:
            y_true_numeric = [float(y) for y in y_true_filtered]
            y_pred_numeric = [float(y) for y in y_pred_filtered]
        except ValueError:
            y_true_numeric = y_true_filtered
            y_pred_numeric = y_pred_filtered

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true_numeric, y_pred_numeric),
            "precision": precision_score(
                y_true_numeric, y_pred_numeric, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true_numeric, y_pred_numeric, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                y_true_numeric, y_pred_numeric, average="weighted", zero_division=0
            ),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true_numeric, y_pred_numeric)
        metrics["confusion_matrix"] = cm.tolist()

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def get_cpds(self) -> Dict[str, Any]:
        """Get Conditional Probability Distributions for all nodes.

        Returns:
            Dictionary mapping node names to their CPDs
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        cpds = {}
        for cpd in self.model.get_cpds():
            cpds[cpd.variable] = {
                "variable": cpd.variable,
                "evidence": cpd.variables[1:] if len(cpd.variables) > 1 else [],
                "values": cpd.values.tolist(),
                "state_names": cpd.state_names,
            }

        return cpds

    def get_model_structure(self) -> Dict[str, Any]:
        """Get information about the model structure.

        Returns:
            Dictionary with model structure information
        """
        if self.model is None:
            self.build_model()

        return {
            "nodes": list(self.model.nodes()),
            "edges": list(self.model.edges()),
            "num_nodes": len(self.model.nodes()),
            "num_edges": len(self.model.edges()),
        }

    def save_model(self, filepath: str) -> None:
        """Save the model to a file.

        Args:
            filepath: Path to save the model
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        try:
            import pickle

            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "structure": self.model_structure,
                        "is_fitted": self._is_fitted,
                    },
                    f,
                )
            logger.success(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    @classmethod
    def load_model(cls, filepath: str) -> "HeartDiseasePredictor":
        """Load a model from a file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded HeartDiseasePredictor instance
        """
        try:
            import pickle

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            predictor = cls(model_structure=data["structure"])
            predictor.model = data["model"]
            predictor._is_fitted = data["is_fitted"]

            if predictor._is_fitted:
                predictor.inference_engine = VariableElimination(predictor.model)

            logger.success(f"Model loaded from {filepath}")
            return predictor

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
