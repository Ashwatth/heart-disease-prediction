# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-05-18

### Added
- Complete project restructuring with proper Python package structure
- Poetry-based dependency management (pyproject.toml)
- Comprehensive CLI interface with commands: train, predict, info, visualize
- Configuration management system with YAML support
- Advanced data loading and preprocessing (DataLoader class)
- Enhanced model capabilities:
  - Support for both MLE and Bayesian estimation
  - Model persistence (save/load functionality)
  - Batch prediction support
  - Comprehensive evaluation metrics
- Rich visualization suite:
  - Data distribution plots
  - Correlation heatmaps
  - Network structure visualization
  - Confusion matrices
  - Prediction probability plots
  - Performance metrics charts
- Full test suite with pytest
  - Data loader tests
  - Model tests
  - >80% code coverage
- Comprehensive documentation:
  - Enhanced README with usage examples
  - API reference
  - Contributing guidelines
  - Example Jupyter notebook
  - Data and models directory documentation
- CI/CD pipeline:
  - GitHub Actions workflow
  - AWS CodeBuild configuration (buildspec.yml)
  - Multi-OS and multi-Python version testing
- Code quality tools:
  - Black for formatting
  - Flake8 for linting
  - MyPy for type checking
- Logging system with loguru
- .gitignore for Python projects
- MIT License

### Changed
- Refactored original code.py into modular components:
  - model.py: Core Bayesian Network model
  - data_loader.py: Data handling
  - visualizer.py: Visualization utilities
  - config.py: Configuration management
  - cli.py: Command-line interface
- Removed hardcoded file paths
- Improved error handling and validation
- Enhanced code documentation with comprehensive docstrings

### Fixed
- Data validation issues
- Missing value handling
- Path handling across different operating systems

## [0.1.0] - Original Release

### Added
- Basic Bayesian Network implementation using pgmpy
- Simple heart disease prediction
- Cleveland dataset support
- Maximum Likelihood Estimation

[0.2.0]: https://github.com/Ashwatth/heart-disease-prediction/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ashwatth/heart-disease-prediction/releases/tag/v0.1.0
