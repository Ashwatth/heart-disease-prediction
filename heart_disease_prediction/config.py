"""Configuration management for the heart disease prediction system."""

from pathlib import Path
from typing import Dict, Any
import yaml
from loguru import logger


class Config:
    """Configuration manager for the application."""

    def __init__(self, config_path: str = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "data": {
                "dataset_path": "data/dataset.csv",
                "test_size": 0.2,
                "random_state": 42,
            },
            "model": {
                "structure": [
                    ["age", "heartdisease"],
                    ["gender", "heartdisease"],
                    ["exang", "heartdisease"],
                    ["cp", "heartdisease"],
                    ["heartdisease", "restecg"],
                    ["heartdisease", "chol"],
                ],
                "target": "heartdisease",
            },
            "logging": {
                "level": "INFO",
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            },
        }

        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        return default_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'data.dataset_path')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: str = None) -> None:
        """Save configuration to file.

        Args:
            path: Path to save configuration. If None, uses original config_path.
        """
        save_path = path or self.config_path
        if not save_path:
            logger.warning("No path specified for saving configuration")
            return

        try:
            with open(save_path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
