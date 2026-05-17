"""Tests for the data loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from heart_disease_prediction.data_loader import DataLoader


@pytest.fixture
def sample_data():
    """Create sample heart disease data for testing."""
    return pd.DataFrame({
        'age': [63, 37, 41, 56, 57],
        'gender': [1, 1, 0, 1, 0],
        'cp': [3, 2, 1, 1, 0],
        'chol': [233, 250, 204, 236, 354],
        'restecg': [1, 1, 0, 1, 1],
        'exang': [0, 0, 0, 0, 0],
        'heartdisease': [0, 0, 0, 0, 0]
    })


@pytest.fixture
def temp_csv_file(sample_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


def test_data_loader_initialization():
    """Test DataLoader initialization."""
    loader = DataLoader()
    assert loader.data is None
    assert loader.train_data is None
    assert loader.test_data is None


def test_load_data_success(temp_csv_file):
    """Test successful data loading."""
    loader = DataLoader()
    data = loader.load_data(temp_csv_file)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 5
    assert loader.data is not None


def test_load_data_file_not_found():
    """Test loading data with non-existent file."""
    loader = DataLoader()
    
    with pytest.raises(FileNotFoundError):
        loader.load_data("nonexistent_file.csv")


def test_preprocess(sample_data):
    """Test data preprocessing."""
    loader = DataLoader()
    loader.data = sample_data.copy()
    
    processed = loader.preprocess()
    
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) == 5


def test_preprocess_with_missing_values():
    """Test preprocessing with missing values."""
    loader = DataLoader()
    data = pd.DataFrame({
        'age': [63, np.nan, 41],
        'gender': [1, 1, 0],
        'heartdisease': [0, 1, 0]
    })
    
    processed = loader.preprocess(data)
    
    # Should drop rows with missing values
    assert len(processed) < len(data)
    assert processed.isnull().sum().sum() == 0


def test_split_data(sample_data):
    """Test data splitting."""
    loader = DataLoader()
    loader.data = sample_data
    
    train, test = loader.split_data(test_size=0.4)
    
    assert len(train) + len(test) == len(sample_data)
    assert len(test) == 2  # 40% of 5
    assert loader.train_data is not None
    assert loader.test_data is not None


def test_get_data_info(sample_data):
    """Test getting data information."""
    loader = DataLoader()
    loader.data = sample_data
    
    info = loader.get_data_info()
    
    assert 'shape' in info
    assert 'columns' in info
    assert 'missing_values' in info
    assert info['shape'] == (5, 7)


def test_validate_data_success(sample_data):
    """Test data validation with valid data."""
    loader = DataLoader()
    
    is_valid = loader.validate_data(sample_data)
    
    assert is_valid is True


def test_validate_data_missing_columns():
    """Test data validation with missing required columns."""
    loader = DataLoader()
    data = pd.DataFrame({
        'age': [63, 37],
        'gender': [1, 1]
    })
    
    is_valid = loader.validate_data(data)
    
    assert is_valid is False
