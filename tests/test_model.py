"""Tests for the model module."""

import pytest
import pandas as pd
import numpy as np
from heart_disease_prediction.model import HeartDiseasePredictor


@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'age': np.random.randint(30, 80, n),
        'gender': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'chol': np.random.randint(150, 350, n),
        'restecg': np.random.randint(0, 3, n),
        'exang': np.random.randint(0, 2, n),
        'heartdisease': np.random.randint(0, 5, n)
    })


@pytest.fixture
def trained_predictor(sample_data):
    """Create and train a predictor."""
    predictor = HeartDiseasePredictor()
    predictor.build_model()
    predictor.fit(sample_data, estimator='mle')
    return predictor


def test_predictor_initialization():
    """Test HeartDiseasePredictor initialization."""
    predictor = HeartDiseasePredictor()
    
    assert predictor.model is None
    assert predictor.inference_engine is None
    assert predictor._is_fitted is False
    assert len(predictor.model_structure) == 6


def test_build_model():
    """Test building Bayesian Network model."""
    predictor = HeartDiseasePredictor()
    model = predictor.build_model()
    
    assert model is not None
    assert predictor.model is not None
    assert len(list(model.nodes())) > 0


def test_fit_mle(sample_data):
    """Test fitting model with Maximum Likelihood Estimation."""
    predictor = HeartDiseasePredictor()
    predictor.fit(sample_data, estimator='mle')
    
    assert predictor._is_fitted is True
    assert predictor.inference_engine is not None


def test_fit_bayes(sample_data):
    """Test fitting model with Bayesian Estimation."""
    predictor = HeartDiseasePredictor()
    predictor.fit(sample_data, estimator='bayes')
    
    assert predictor._is_fitted is True


def test_predict(trained_predictor):
    """Test making a single prediction."""
    evidence = {'age': 50, 'gender': 1, 'cp': 2}
    
    prediction = trained_predictor.predict(evidence)
    
    assert isinstance(prediction, dict)
    assert len(prediction) > 0
    # Check probabilities sum to ~1
    assert abs(sum(prediction.values()) - 1.0) < 0.01


def test_predict_not_fitted():
    """Test prediction before fitting raises error."""
    predictor = HeartDiseasePredictor()
    evidence = {'age': 50}
    
    with pytest.raises(RuntimeError):
        predictor.predict(evidence)


def test_predict_batch(trained_predictor, sample_data):
    """Test batch predictions."""
    test_data = sample_data.head(10)
    
    predictions = trained_predictor.predict_batch(test_data)
    
    assert len(predictions) == 10
    assert all(isinstance(pred, dict) for pred in predictions)


def test_evaluate(trained_predictor, sample_data):
    """Test model evaluation."""
    test_data = sample_data.head(30)
    
    metrics = trained_predictor.evaluate(test_data)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_get_cpds(trained_predictor):
    """Test getting Conditional Probability Distributions."""
    cpds = trained_predictor.get_cpds()
    
    assert isinstance(cpds, dict)
    assert len(cpds) > 0


def test_get_model_structure():
    """Test getting model structure information."""
    predictor = HeartDiseasePredictor()
    
    structure = predictor.get_model_structure()
    
    assert 'nodes' in structure
    assert 'edges' in structure
    assert structure['num_nodes'] > 0
    assert structure['num_edges'] == 6


def test_save_and_load_model(trained_predictor, tmp_path):
    """Test saving and loading model."""
    model_path = tmp_path / "test_model.pkl"
    
    # Save model
    trained_predictor.save_model(str(model_path))
    assert model_path.exists()
    
    # Load model
    loaded_predictor = HeartDiseasePredictor.load_model(str(model_path))
    
    assert loaded_predictor._is_fitted is True
    assert loaded_predictor.model is not None
    
    # Test prediction with loaded model
    evidence = {'age': 50, 'gender': 1, 'cp': 2}
    prediction = loaded_predictor.predict(evidence)
    assert isinstance(prediction, dict)
