# Heart Disease Risk Prediction

## Overview
This project focuses on predicting whether a patient has heart disease based on clinical and diagnostic attributes.
Instead of chasing the highest possible accuracy, the goal here was to build a clean, **interpretable baseline model** and understand *why* it behaves the way it does — which is especially important in healthcare-related problems.

-----

## Why this project?
Heart disease diagnosis is a high-stakes problem. In such domains, blindly optimizing metrics can be misleading.
A model that is easy to explain and has a strong recall (fewer missed positive cases) is oftem more valuable then a slightly more accurate but opaque model.

This project was built with that mindset.

-----

## Dataset
* **Source**: Kaggle – Heart Disease Prediction dataset
* **Samples**: 270
* **Features**: 13 input features + 1 target
* **Target**: `Heart Disease`
    * `0` -> Absence
    * `1` -> Presence

The dataset contains a mix of numerical feature and categorical variables that are already encoded as integers (e.g., chest pain type, exercise angina).

-----

## What I did

### 1. Exploratory Data Analysis
* Identified the target variable and checked class balance
* Inspected features types and noted encoded categorical variables
* Verified that the dataset has no missing values

This helped confirm that the data was clean and suitable for classical machine learning models.

### 2. Baseline Model
I started with a `Logistic Regression` model as a baseline.
* Train-test split: **80/20 (stratified)**
* Metrics used:
    * Accuracy
    * Confusion matrix
    * Precision, Recall, F1-score

Rather than focusing only on accuracy, I paid close attention to **recall for the positive class**, since missing a heart disease case is more costly than a false alarm.

### 3. Feature Scaling
Since Logistic Regression relies on gradient-based optimization, I applied `StandardScaler`:

* Scaler fitted only on training data (to avoid data leakage)

* Retrained the model on scaled features

This:

* Removed convergence warnings

* Improved optimization stability

* Made coefficient magnitudes more meaningful

Scaling didn’t dramatically change accuracy, but it made the model behave more reliably.

### 4. Regularization & Interpretability
I compared:

* L2 (Ridge) Logistic Regression

* L1 (Lasso) Logistic Regression

Key observations:

* L2 kept all features but shrank their coefficients smoothly

* L1 introduced sparsity and eliminated weak predictors

* `Age` was driven to zero by L1, suggesting it adds little predictive value once other features are considered

* Features like `number of vessels fluro`, `chest pain type`, and `sex` were consistently important across both models

This comparison highlighted the trade-off between **stability (L2) and interpretability (L1)**.

-----

## Final Model Choice

**Logistic Regression with L2 regularization on scaled features.**

Why this model?
* Stable coefficients
* Strong recall
* Interpretable behavior
* Well-suited for healthcare use cases

This version is intentionally kept simple and explainable.

-----

## Result (High-level)

* Accuracy: ~85-88% (depending on split)
* Strong recall for heart disease cases
* Feature importance aligns with medical intuition

Accuracy was **not** treated as the only success metric — reducing false negatives was the priority.

The final trained model is saved for reuse and inference.

## Project Structure
```
heart-disease-risk-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_scaled_model.ipynb
│   └── 04_regularization_analysis.ipynb
├── src/
├── models/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore

```

## Future Improvements
This project is frozen as v1.0. Possible next steps include:
* Threshold tuning to better control recall vs precision
* ROC-AUC analysis
* Comparing against tree-based models (Random Forest, XGBoost)
* Deploying the model as a simple prediction API

## Key Takeaways
* A strong baseline is more valuable than rushed optimization
* Feature scaling is essential for gradient-based models
* Regularization affects interpretability more than raw accuracy
* In healtcare ML, **explainability matters as much as performance**

## How to Run

```
pip install -r requirements.txt
```

### Run notebooks in order:
`01_eda → 02_baseline_model → 03_scaled_model → 04_regularization_analysis
`