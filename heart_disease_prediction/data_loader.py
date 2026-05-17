"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from loguru import logger
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handle data loading and preprocessing for heart disease prediction."""

    def __init__(self, config: dict = None):
        """Initialize the data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load heart disease dataset from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with shape {data.shape} from {filepath}")
            
            # Replace missing value indicators
            data = data.replace("?", np.nan)
            
            self.data = data
            return data

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise ValueError(f"Invalid data format: {e}")

    def preprocess(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Preprocess the data.

        Args:
            data: DataFrame to preprocess. If None, uses self.data

        Returns:
            Preprocessed DataFrame
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available for preprocessing")
            data = self.data.copy()
        else:
            data = data.copy()

        # Handle missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            # Drop rows with missing values for now
            data = data.dropna()
            logger.info(f"After dropping missing values, shape: {data.shape}")

        # Convert data types if needed
        for col in data.columns:
            if data[col].dtype == "object":
                try:
                    data[col] = pd.to_numeric(data[col])
                except ValueError:
                    logger.warning(f"Could not convert column {col} to numeric")

        return data

    def split_data(
        self, data: pd.DataFrame = None, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets.

        Args:
            data: DataFrame to split. If None, uses self.data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, test_data)
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available for splitting")
            data = self.data

        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        self.train_data = train_data
        self.test_data = test_data

        logger.info(f"Split data into train ({len(train_data)}) and test ({len(test_data)}) sets")
        return train_data, test_data

    def get_data_info(self, data: pd.DataFrame = None) -> dict:
        """Get information about the dataset.

        Args:
            data: DataFrame to analyze. If None, uses self.data

        Returns:
            Dictionary with dataset information
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data available")
            data = self.data

        info = {
            "shape": data.shape,
            "columns": list(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "dtypes": data.dtypes.to_dict(),
            "description": data.describe().to_dict(),
        }

        return info

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that the data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required_columns = ["age", "gender", "cp", "chol", "restecg", "exang", "heartdisease"]
        
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        logger.info("Data validation passed")
        return True
