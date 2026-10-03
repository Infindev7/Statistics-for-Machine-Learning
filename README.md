# Statistical Methods & Machine Learning Experiments (1–9)

A comprehensive collection of 9 statistical and machine learning practical experiments implemented in Python and Jupyter Notebooks. The series systematically explores data analysis, hypothesis testing, predictive modeling, clustering, and model interpretability using the classic **Pima Indians Diabetes Dataset**.

---

## 📚 Experiment Overview

| Exp # | Experiment Title | Core Concepts & Techniques | Key Outputs & Visualizations |
|:---:|:---|:---|:---|
| **01** | **Exploratory Statistical Analysis** | Data summary statistics, missing-value identification (zero-imputation), variable distributions, correlation heatmap | `fig1_outcome_dist.png`<br>`fig2_histograms.png`<br>`fig3_boxplots.png`<br>`fig4_correlation.png` |
| **02** | **Descriptive Statistics & Data Visualization** | Measures of central tendency (mean, median), dispersion (std, IQR), skewness, kurtosis, group comparisons | `fig1_dist_by_outcome.png`<br>`fig2_glucose_box.png`<br>`fig3_bmi_violin.png` |
| **03** | **Correlation Analysis & Distance Measures** | Pearson & Spearman correlation, multi-collinearity, feature standardization vs min-max scaling, Euclidean distance | `fig1_corr.png`<br>`fig2_distances.png` |
| **04** | **Statistical Inference & Hypothesis Testing** | Point estimation, standard errors, 95% Confidence Intervals, two-sample independent t-test, Mann-Whitney U test, Chi-Square test | `fig1_means_ci.png` |
| **05** | **Resampling Techniques & CI Estimation** | Non-parametric bootstrap, bootstrap sampling distribution, percentile & pivotal CIs, permutation testing | `fig1_bootstrap.png` |
| **06** | **Regression Models & Statistical Measures** | Ordinary Least Squares (OLS), Ridge (L2), Lasso (L1) regression, MSE, RMSE, MAE, R², residual diagnostics | `fig1_regression.png` |
| **07** | **Classification Models (Supervised Learning)** | Logistic Regression, Random Forest Classifier, Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC | `fig1_roc.png` |
| **08** | **Clustering & Dimensionality Reduction** | Principal Component Analysis (PCA), explained variance ratio, K-Means clustering, Elbow method, Silhouette score | `fig1_pca.png`<br>`fig2_elbow.png`<br>`fig3_kmeans.png` |
| **09** | **Model Validation & Explainability** | Stratified K-Fold cross-validation, probability calibration curve, Brier score, Permutation Feature Importance | `fig1_calibration.png`<br>`fig2_importance.png` |

---

## 📊 Dataset Description

All experiments utilize the **Pima Indians Diabetes Dataset** (`pima-indians-diabetes.csv`):
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases (UCI Machine Learning Repository).
- **Instances**: 768 patient records.
- **Attributes**: 8 numeric predictor variables + 1 binary target outcome:
  1. `Pregnancies`: Number of times pregnant
  2. `Glucose`: Plasma glucose concentration after a 2-hour oral glucose tolerance test
  3. `BloodPressure`: Diastolic blood pressure ($mm\ Hg$)
  4. `SkinThickness`: Triceps skin fold thickness ($mm$)
  5. `Insulin`: 2-hour serum insulin ($mu\ U/ml$)
  6. `BMI`: Body mass index ($weight\ in\ kg / (height\ in\ m)^2$)
  7. `DiabetesPedigreeFunction`: Genetic diabetes score
  8. `Age`: Patient age (years)
  9. `Outcome`: Class variable (`0` = Non-diabetic, `1` = Diabetic)
- **Data Quality Note**: In biologically impossible cases, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing data and appropriately handled across experiments.

---

## 🛠️ Requirements & Installation

Python 3.9+ is recommended. Install required libraries with pip:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
```

---

## 🚀 How to Run the Experiments

1. Navigate to the desired experiment directory (e.g., `Expt 1/`).
2. Launch Jupyter Notebook / JupyterLab or open in VS Code:
   ```bash
   jupyter notebook Expt_1.ipynb
   ```
3. To recompute all results and save figures: Select **Kernel > Restart & Run All**.
4. Generated charts will be saved directly into each experiment's `images/` directory.

---

## 📂 Repository Structure

```
Stats/
├── .gitignore
├── README.md
├── Expt 1/
│   ├── Expt_1.ipynb
│   ├── Expt_1.html
│   ├── pima-indians-diabetes.csv
│   ├── README.md
│   └── images/
├── Expt 2/
│   └── ...
├── ...
└── Expt 9/
    ├── Expt_9.ipynb
    ├── Expt_9.html
    ├── pima-indians-diabetes.csv
    ├── README.md
    └── images/
```
