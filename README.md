# Heart Disease Prediction System 🫀

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Bayesian Network-based system for heart disease prediction with advanced features including data visualization, model evaluation, and a user-friendly CLI.

## 🌟 Features

### Core Capabilities
- **Bayesian Network Modeling**: Probabilistic graphical model for heart disease prediction
- **Multiple Estimation Methods**: Support for Maximum Likelihood Estimation (MLE) and Bayesian Estimation
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Data Validation**: Automatic validation and preprocessing of input data
- **Flexible Configuration**: YAML-based configuration management

### Enhanced Features (New in v0.2.0)
- 🎨 **Rich Visualizations**: Network structure, data distribution, correlation heatmaps, and more
- 🖥️ **Command-Line Interface**: Easy-to-use CLI for training, prediction, and analysis
- 📊 **Advanced Analytics**: Detailed data profiling and model performance metrics
- 🧪 **Comprehensive Testing**: Full test suite with pytest
- 📦 **Poetry Integration**: Modern dependency management
- 🔍 **Logging System**: Detailed logging with loguru
- 💾 **Model Persistence**: Save and load trained models
- 🔄 **Batch Predictions**: Process multiple samples at once

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Making Predictions](#making-predictions)
  - [Dataset Information](#dataset-information)
  - [Visualizations](#visualizations)
- [Dataset](#dataset)
- [Bayesian Networks](#bayesian-networks)
- [API Reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/Ashwatth/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/Ashwatth/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -e .
```

## 🎯 Quick Start

### 1. Prepare Your Data

Place your heart disease dataset (CSV format) in the `data/` directory. The dataset should include columns:
- `age`: Patient age
- `gender`: Gender (0: female, 1: male)
- `cp`: Chest pain type (0-3)
- `chol`: Serum cholesterol in mg/dl
- `restecg`: Resting electrocardiographic results (0-2)
- `exang`: Exercise induced angina (0: no, 1: yes)
- `heartdisease`: Target variable (0-4, 0 = no disease)

### 2. Train a Model

```bash
heart-predict train --data data/dataset.csv --visualize --save-model models/heart_model.pkl
```

### 3. Make Predictions

```bash
heart-predict predict --model models/heart_model.pkl --age 55 --gender 1 --cp 2 --chol 250 --visualize
```

## 📖 Usage

### Training a Model

Train a model with custom parameters:

```bash
heart-predict train \
  --data data/dataset.csv \
  --output results/ \
  --test-size 0.3 \
  --estimator bayes \
  --save-model models/my_model.pkl \
  --visualize
```

**Options:**
- `--data`, `-d`: Path to dataset CSV file (required)
- `--config`, `-c`: Path to configuration file (optional)
- `--output`, `-o`: Output directory for results (default: `output`)
- `--test-size`: Test set proportion (default: 0.2)
- `--estimator`: Estimation method - `mle` or `bayes` (default: `mle`)
- `--save-model`: Path to save the trained model
- `--visualize`: Generate visualizations

### Making Predictions

Predict heart disease for a single patient:

```bash
heart-predict predict \
  --model models/heart_model.pkl \
  --age 63 \
  --gender 1 \
  --cp 3 \
  --chol 233 \
  --restecg 1 \
  --exang 0 \
  --visualize
```

**Required Options:**
- `--model`, `-m`: Path to trained model file
- `--age`: Patient age
- `--gender`: Gender (0 or 1)
- `--cp`: Chest pain type (0-3)

**Optional Options:**
- `--chol`: Cholesterol level
- `--restecg`: Resting ECG results (0-2)
- `--exang`: Exercise induced angina (0 or 1)
- `--visualize`: Show probability visualization

### Dataset Information

View detailed information about your dataset:

```bash
heart-predict info --data data/dataset.csv
```

### Visualizations

Generate comprehensive visualizations:

```bash
heart-predict visualize --data data/dataset.csv --output visualizations/
```

## 📊 Dataset

### Cleveland Heart Disease Database

The system is designed for the Cleveland Heart Disease Database, which contains 303 instances with 14 attributes:

| Attribute | Description | Values |
|-----------|-------------|--------|
| age | Age in years | Numeric |
| gender | Sex | 0 = female, 1 = male |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina | 0 = no, 1 = yes |
| oldpeak | ST depression | Numeric |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels | 0-3 |
| thal | Thalassemia | 1-3 |
| heartdisease | Target (diagnosis) | 0-4 (0 = no disease) |

**Database Distribution:**
- No disease (0): 164 instances
- Disease levels 1-4: 139 instances total

## 🧠 Bayesian Networks

### What is a Bayesian Network?

A Bayesian Network is a probabilistic graphical model that represents variables and their conditional dependencies via a directed acyclic graph (DAG).

### Model Structure

The default model structure captures the following relationships:

```
      age ─┐
   gender ─┤
    exang ─┼──> heartdisease ──┬──> restecg
       cp ─┘                    └──> chol
```

### Probability Calculations

The joint probability distribution is computed as:

```
P(U) = ∏ P(Ai | pa(Ai))
```

Where:
- `U` is the set of all variables
- `Ai` are individual variables
- `pa(Ai)` are the parent nodes of Ai

### Inference

The system uses Variable Elimination for inference:

```
P(Disease | Evidence) = P(Disease, Evidence) / P(Evidence)
```

## 🔧 API Reference

### Python API

```python
from heart_disease_prediction import HeartDiseasePredictor, DataLoader, Visualizer

# Load and preprocess data
loader = DataLoader()
data = loader.load_data("data/dataset.csv")
processed_data = loader.preprocess(data)
train_data, test_data = loader.split_data(processed_data)

# Build and train model
predictor = HeartDiseasePredictor()
predictor.fit(train_data, estimator='mle')

# Make prediction
evidence = {'age': 55, 'gender': 1, 'cp': 2}
prediction = predictor.predict(evidence)
print(f"Prediction: {prediction}")

# Evaluate model
metrics = predictor.evaluate(test_data)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Visualize
viz = Visualizer()
viz.set_output_dir("visualizations")
viz.plot_network_structure(predictor.model_structure, save=True)
```

## 👩‍💻 Development

### Setup Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Activate pre-commit hooks (if configured)
pre-commit install

# Format code
poetry run black heart_disease_prediction tests

# Run linter
poetry run flake8 heart_disease_prediction

# Type checking
poetry run mypy heart_disease_prediction
```

### Project Structure

```
heart-disease-prediction/
├── heart_disease_prediction/     # Main package
│   ├── __init__.py
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration management
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── model.py                 # Bayesian Network model
│   └── visualizer.py            # Visualization utilities
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_model.py
├── data/                        # Dataset directory
├── config.example.yaml          # Example configuration
├── pyproject.toml              # Poetry configuration
├── README.md                   # This file
└── .gitignore
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=heart_disease_prediction --cov-report=html

# Run specific test file
poetry run pytest tests/test_model.py

# Run with verbose output
poetry run pytest -v
```

## 📈 Performance

Typical performance on the Cleveland dataset:
- **Accuracy**: 75-85%
- **Training Time**: < 1 second
- **Prediction Time**: < 0.01 seconds per sample

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide
- Use Black for code formatting
- Add tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Cleveland Heart Disease Database from UCI Machine Learning Repository
- pgmpy library for Bayesian Network implementation
- The open-source community for excellent tools and libraries

## 📧 Contact

**Ashwatth**
- GitHub: [@Ashwatth](https://github.com/Ashwatth)

## 🗺️ Roadmap

- [ ] Add support for more datasets
- [ ] Implement structure learning algorithms
- [ ] Web interface for predictions
- [ ] Docker containerization
- [ ] REST API
- [ ] Real-time monitoring dashboard
- [ ] Integration with medical databases
- [ ] Mobile application

---

**Made with ❤️ for better healthcare through AI**
