# Experiment 1: Exploratory Statistical Analysis

## Aim
To perform exploratory statistical analysis of the Pima Indians Diabetes Dataset by examining its structure, variable types, data quality, statistical characteristics and important patterns.

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
1. Open `Expt_1.ipynb` in Jupyter Notebook / JupyterLab / VS Code, or upload it to Google Colab (also upload `pima-indians-diabetes.csv` alongside it in Colab).
2. Run all cells: **Kernel > Restart & Run All**.
3. Figures are saved automatically to the `images/` folder when the notebook runs.

## Outputs
The notebook prints statistics, test results and model metrics inline, and saves these figures to `images/`:
- `fig1_outcome_dist.png` - Bar chart of diabetes outcome distribution (0 = No, 1 = Yes).
- `fig2_histograms.png` - Histograms of all eight input variables.
- `fig3_boxplots.png` - Boxplots of input variables for outlier detection.
- `fig4_correlation.png` - Correlation heatmap of all variables.
- `notebook_screenshot.png` - full-page screenshot of the executed notebook (code + outputs).

## Files in this folder
- `Expt_1.ipynb` - the executed notebook (code with outputs).
- `pima-indians-diabetes.csv` - the dataset used.
- `images/` - output figures and the notebook screenshot.
- `README.md` - this file.
