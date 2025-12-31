# Solar Power Prediction

Notebook Version by my pair Omar Elnazly: https://github.com/Omar-Elnazly/Aswan-Weather-Analytics

This repository contains a comprehensive pipeline for predicting solar power output using meteorological data. It performs data cleaning, extensive feature engineering, statistical analysis, and machine learning modeling (both classification and regression).

## Features

### 1. Data Processing
- **Cleaning**: Handles missing values via mean imputation and removes outliers using the IQR method.
- **Feature Engineering**:
  - **Time Features**: Extracts Month, Day of Year, and Week of Year.
  - **Interaction Terms**: Generates features like `Temp * Humidity`, `Wind * Pressure`.
  - **Polynomial Features**: Adds squared terms for Temperature, Wind, and DewPoint to capture non-linear relationships.

### 2. Statistical Analysis
- **EDA**: Generates descriptive statistics, correlation/covariance matrices, and distribution plots.
- **Hypothesis Testing**:
  - **ANOVA**: Tests variance of Temperature across SolarPV classes.
  - **T-Test**: Compares Solar output between high and low humidity groups.
  - **Chi-Square**: Examines independence between Temperature categories and SolarPV classes.

### 3. Machine Learning Models
The pipeline applies dimensionality reduction (PCA, LDA, SVD, KernelPCA) before training models.

#### Classification
Predicts solar power generation as `Low`, `Medium`, or `High`.
- **Models**: Naive Bayes, Decision Tree, KNN (Euclidean, Manhattan, Minkowski), Linear Discriminant Analysis (LDA), Logistic Regression, Neural Network (MLP), and a simplified Bayesian Network.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curves, and Learning Curves.

#### Regression
Predicts the exact solar power output value.
- **Models**: Linear Regression, Neural Network Regressor.
- **Metrics**: MAE, RMSE, R², Willmott Index, Nash-Sutcliffe Efficiency (NSE).

## Usage

1. **Setup Environment**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Analysis**:
   ```sh
   python main.py
   ```
   All results (CSV summaries and PNG plots) will be generated in the `output/` directory.
