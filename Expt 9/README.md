# Experiment 9: Model Validation and Explainability

## Aim
To evaluate and interpret machine learning models using cross-validation, calibration and explainability techniques.

## Dataset
**Pima Indians Diabetes Dataset** (`pima-indians-diabetes.csv`, included in this folder)

- Source: National Institute of Diabetes and Digestive and Kidney Diseases (public mirror of the classic UCI dataset).
- 768 patient records, 8 numeric input features + 1 binary outcome.
- Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome (1 = diabetic, 0 = not diabetic).
- Data-quality note: zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are biologically invalid and treated as missing during preprocessing.

## Requirements
- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter (notebook)

Install everything with:
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to run
1. Open `Expt_9.ipynb` in Jupyter Notebook / JupyterLab / VS Code, or upload it to Google Colab (also upload `pima-indians-diabetes.csv` alongside it in Colab).
2. Run all cells: **Kernel > Restart & Run All**.
3. Figures are saved automatically to the `images/` folder when the notebook runs.

## Outputs
The notebook prints statistics, test results and model metrics inline, and saves these figures to `images/`:
- `fig1_calibration.png` - Calibration curve of the Random Forest classifier.
- `fig2_importance.png` - Permutation feature importance (mean decrease in accuracy).
- `notebook_screenshot.png` - full-page screenshot of the executed notebook (code + outputs).

## Files in this folder
- `Expt_9.ipynb` - the executed notebook (code with outputs).
- `pima-indians-diabetes.csv` - the dataset used.
- `images/` - output figures and the notebook screenshot.
- `README.md` - this file.
