# Contributing to Heart Disease Prediction System

Thank you for your interest in contributing! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git`
3. Add upstream remote: `git remote add upstream https://github.com/Ashwatth/heart-disease-prediction.git`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management

### Installation

```bash
# Install Poetry if you haven't already
pip install poetry

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=heart_disease_prediction --cov-report=html

# Run specific test
pytest tests/test_model.py
```

### Code Quality Tools

```bash
# Format code
black heart_disease_prediction tests

# Lint code
flake8 heart_disease_prediction tests

# Type checking
mypy heart_disease_prediction
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add usage examples or tutorials

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use Black for code formatting

### Documentation

- Add docstrings to all public functions, classes, and methods
- Use Google-style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

### Naming Conventions

- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Git Commit Messages

Write clear and descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**

```
feat: Add batch prediction support

- Implement predict_batch method in HeartDiseasePredictor
- Add tests for batch predictions
- Update CLI to support batch input files

Closes #123
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Use fixtures for common setup

```python
def test_predictor_makes_valid_prediction():
    """Test that predictor returns valid probability distribution."""
    predictor = HeartDiseasePredictor()
    # ... test implementation
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test both success and failure paths

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** and ensure they pass:
   ```bash
   pytest
   ```

3. **Check code quality**:
   ```bash
   black heart_disease_prediction tests
   flake8 heart_disease_prediction tests
   mypy heart_disease_prediction
   ```

4. **Update documentation** if needed

5. **Add tests** for new functionality

### Submitting the PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template with:
   - Clear description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)

### PR Requirements

- All tests must pass
- Code coverage should not decrease
- Code must pass linting
- Documentation must be updated
- Commit messages should be clear

### Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, a maintainer will merge your PR

## Reporting Bugs

### Before Submitting a Bug Report

1. Check if the bug has already been reported
2. Try to reproduce the bug with the latest version
3. Collect relevant information:
   - Python version
   - Operating system
   - Error messages/stack traces
   - Steps to reproduce

### Submitting a Bug Report

Create an issue with:

- **Clear title**: Brief description of the bug
- **Description**: Detailed explanation
- **Steps to reproduce**: Numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, etc.
- **Additional context**: Screenshots, logs, etc.

**Example:**

```markdown
## Bug: Model fails to load from saved file

### Description
When trying to load a saved model using `load_model()`, a `pickle.UnpicklingError` is raised.

### Steps to Reproduce
1. Train a model
2. Save using `predictor.save_model('model.pkl')`
3. Load using `HeartDiseasePredictor.load_model('model.pkl')`

### Expected Behavior
Model should load successfully

### Actual Behavior
Raises: `pickle.UnpicklingError: invalid load key`

### Environment
- Python 3.10.5
- Windows 11
- Package version: 0.2.0

### Additional Context
Error only occurs on Windows, works fine on Linux.
```

## Suggesting Features

### Before Suggesting a Feature

1. Check if the feature has already been suggested
2. Ensure it aligns with project goals
3. Consider if it would be useful to most users

### Submitting a Feature Request

Create an issue with:

- **Clear title**: Brief description of the feature
- **Problem statement**: What problem does it solve?
- **Proposed solution**: How would it work?
- **Alternatives**: Other approaches considered
- **Use cases**: When would this be used?

## Development Workflow

### Typical Workflow

1. **Check existing issues** or create a new one
2. **Discuss the approach** before major changes
3. **Create a branch** from `main`
4. **Make changes** following coding standards
5. **Add tests** for your changes
6. **Update documentation**
7. **Commit changes** with clear messages
8. **Push to your fork**
9. **Create a Pull Request**
10. **Address review feedback**

### Branch Naming

- Feature: `feature/feature-name`
- Bug fix: `fix/bug-description`
- Documentation: `docs/what-changed`
- Refactor: `refactor/what-changed`

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Check existing documentation

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README

Thank you for contributing! 🎉
