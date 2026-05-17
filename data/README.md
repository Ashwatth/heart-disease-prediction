# Data Directory

Place your heart disease dataset CSV files here.

## Expected Format

The CSV file should contain the following columns:

- `age`: Patient age (numeric)
- `gender`: Gender (0 = female, 1 = male)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure in mm Hg (optional)
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl (optional)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved (optional)
- `exang`: Exercise induced angina (0 = no, 1 = yes)
- `oldpeak`: ST depression induced by exercise (optional)
- `slope`: Slope of the peak exercise ST segment (optional)
- `ca`: Number of major vessels colored by fluoroscopy (optional)
- `thal`: Thalassemia (optional)
- `heartdisease`: Target variable (0-4, where 0 = no disease)

## Example

```csv
age,gender,cp,chol,restecg,exang,heartdisease
63,1,3,233,1,0,0
37,1,2,250,1,0,0
41,0,1,204,0,0,0
```

## Data Sources

The Cleveland Heart Disease Database is available from:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
