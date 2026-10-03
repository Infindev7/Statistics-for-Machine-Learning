# Experiment 8: Clustering and Dimensionality Reduction for Pattern Discovery

## Aim
To apply clustering and dimensionality reduction techniques for unsupervised pattern discovery.

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
1. Open `Expt_8.ipynb` in Jupyter Notebook / JupyterLab / VS Code, or upload it to Google Colab (also upload `pima-indians-diabetes.csv` alongside it in Colab).
2. Run all cells: **Kernel > Restart & Run All**.
3. Figures are saved automatically to the `images/` folder when the notebook runs.

## Outputs
The notebook prints statistics, test results and model metrics inline, and saves these figures to `images/`:
- `fig1_pca.png` - PCA projection (2 components) colored by true outcome.
- `fig2_elbow.png` - Elbow method plot for choosing k in K-Means.
- `fig3_kmeans.png` - K-Means clusters (k=2) on the PCA projection.
- `notebook_screenshot.png` - full-page screenshot of the executed notebook (code + outputs).

## Files in this folder
- `Expt_8.ipynb` - the executed notebook (code with outputs).
- `pima-indians-diabetes.csv` - the dataset used.
- `images/` - output figures and the notebook screenshot.
- `README.md` - this file.
