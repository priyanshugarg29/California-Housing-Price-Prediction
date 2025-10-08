# California Housing Price Prediction

This repository contains a complete data science workflow for predicting median house values in California districts, using advanced feature engineering, regularization, and robust model evaluation.

## Overview

The project applies machine learning to the California Housing dataset. It includes:
- Data cleaning and preprocessing
- Feature transformation (log1p, capping)
- Handling multicollinearity
- Ridge regression with hyperparameter tuning
- Residual analysis and model diagnostics
- Interpretability and critical evaluation

## Key Steps

1. **Data Preprocessing**
   - Outlier treatment and feature capping (notably for Population)
   - Feature scaling and log-transformations to stabilize variance and address skewness

2. **Feature Selection**
   - Exploratory data analysis to identify key predictors
   - Addressing multicollinearity through VIF and regularization

3. **Modeling**
   - Ridge Regression with cross-validated alpha
   - Comparison of transformed vs untransformed/capped features
   - Performance measured on separate train/test splits

4. **Diagnostics**
   - Residuals plotted for normality and heteroscedasticity
   - Metrics reported: R-squared, MSE, RMSE, MAE

5. **Critical Evaluation**
   - Discussion of strengths, limitations, and practical acceptability in real-world contexts
   - Recommendations for further improvements (non-linear models, richer features)

## Results

- **Best model:** Ridge regression with log1p-transformed features and capped population
- **Performance (test set):**
  - R² ≈ 0.666
  - MSE ≈ 0.0414
  - RMSE ≈ 0.2034
  - MAE ≈ 0.1544
- **Conclusion:** The model provides robust baseline predictions suitable for practical exploratory use, with further gains possible through non-linear modeling and richer data sources.

## Getting Started

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Run the main notebook or script:
