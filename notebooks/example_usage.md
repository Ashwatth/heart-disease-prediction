# Example Jupyter Notebook for Heart Disease Prediction

This notebook demonstrates the usage of the heart disease prediction system.

## Installation

```python
# Install required packages (if not already installed)
# !pip install numpy pandas pgmpy scikit-learn matplotlib seaborn
```

## Import Libraries

```python
import sys
sys.path.append('..')

from heart_disease_prediction import HeartDiseasePredictor, DataLoader, Visualizer
import pandas as pd
import numpy as np

# Set display options
pd.set_option('display.max_columns', None)
%matplotlib inline
```

## Load and Explore Data

```python
# Initialize data loader
loader = DataLoader()

# Load dataset
data = loader.load_data('../data/dataset.csv')

# Display basic info
print("Dataset Shape:", data.shape)
print("\nFirst few rows:")
data.head()
```

## Data Preprocessing

```python
# Get data information
info = loader.get_data_info(data)
print("Missing Values:")
for col, missing in info['missing_values'].items():
    if missing > 0:
        print(f"  {col}: {missing}")

# Preprocess data
processed_data = loader.preprocess(data)
print(f"\nProcessed data shape: {processed_data.shape}")

# Split into train and test
train_data, test_data = loader.split_data(processed_data, test_size=0.2)
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")
```

## Visualize Data

```python
# Create visualizer
viz = Visualizer()
viz.set_output_dir('../visualizations')

# Plot data distribution
viz.plot_data_distribution(processed_data, save=False)

# Plot feature correlation
viz.plot_feature_correlation(processed_data, save=False)
```

## Build and Train Model

```python
# Initialize predictor
predictor = HeartDiseasePredictor()

# Build model
predictor.build_model()

# Get model structure
structure = predictor.get_model_structure()
print("Model Structure:")
print(f"  Nodes: {structure['num_nodes']}")
print(f"  Edges: {structure['num_edges']}")
print(f"\nEdges:")
for edge in structure['edges']:
    print(f"  {edge[0]} -> {edge[1]}")

# Train model
print("\nTraining model...")
predictor.fit(train_data, estimator='mle')
print("Training complete!")
```

## Visualize Network Structure

```python
viz.plot_network_structure(predictor.model_structure, save=False)
```

## Make Predictions

```python
# Single prediction
evidence = {
    'age': 55,
    'gender': 1,
    'cp': 2,
    'chol': 250,
    'restecg': 1,
    'exang': 0
}

prediction = predictor.predict(evidence)
print("Prediction for patient:")
for key, value in evidence.items():
    print(f"  {key}: {value}")
    
print("\nPredicted probabilities:")
for level, prob in sorted(prediction.items()):
    print(f"  Level {level}: {prob:.4f} ({prob*100:.2f}%)")

# Visualize prediction
viz.plot_prediction_probabilities(prediction, save=False)
```

## Batch Predictions

```python
# Predict on test set
predictions = predictor.predict_batch(test_data.head(10))

# Display results
for i, pred in enumerate(predictions):
    if pred:
        most_likely = max(pred.items(), key=lambda x: x[1])
        print(f"Sample {i+1}: Level {most_likely[0]} ({most_likely[1]*100:.1f}%)")
```

## Model Evaluation

```python
# Evaluate on test set
metrics = predictor.evaluate(test_data)

print("Model Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1 Score:  {metrics['f1_score']:.4f}")

# Visualize metrics
viz.plot_metrics(metrics, save=False)

# Confusion matrix
if 'confusion_matrix' in metrics:
    cm = np.array(metrics['confusion_matrix'])
    viz.plot_confusion_matrix(cm, save=False)
```

## Get Conditional Probability Distributions

```python
# Get CPDs
cpds = predictor.get_cpds()

# Display CPD for heartdisease
if 'heartdisease' in cpds:
    cpd = cpds['heartdisease']
    print(f"CPD for {cpd['variable']}:")
    print(f"  Evidence: {cpd['evidence']}")
    print(f"  State names: {cpd['state_names']}")
```

## Save Model

```python
# Save the trained model
model_path = '../models/trained_model.pkl'
predictor.save_model(model_path)
print(f"Model saved to {model_path}")
```

## Load and Use Saved Model

```python
# Load model
loaded_predictor = HeartDiseasePredictor.load_model(model_path)

# Make prediction with loaded model
test_evidence = {'age': 60, 'gender': 0, 'cp': 1}
loaded_prediction = loaded_predictor.predict(test_evidence)

print("Prediction with loaded model:")
for level, prob in sorted(loaded_prediction.items()):
    print(f"  Level {level}: {prob:.4f}")
```

## Advanced: Compare Estimators

```python
# Train with different estimators
predictors = {}

for estimator in ['mle', 'bayes']:
    print(f"\nTraining with {estimator.upper()}...")
    pred = HeartDiseasePredictor()
    pred.fit(train_data, estimator=estimator)
    metrics = pred.evaluate(test_data)
    predictors[estimator] = {
        'predictor': pred,
        'metrics': metrics
    }
    print(f"  Accuracy: {metrics['accuracy']:.4f}")

# Compare results
print("\nComparison:")
for est, data in predictors.items():
    print(f"{est.upper()}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        print(f"  {metric}: {data['metrics'][metric]:.4f}")
```

## Conclusion

This notebook demonstrated:
- Loading and preprocessing heart disease data
- Training Bayesian Network models
- Making predictions
- Evaluating model performance
- Visualizing results
- Saving and loading models

For more information, see the [README](../README.md).
