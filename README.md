APS Failure Prediction using Feature Selection and Ensemble Learning

This notebook addresses the classification of APS (Air Pressure System) failure data using imputation, feature engineering, and ensemble classifiers such as Random Forest and XGBoost.

## Objectives
- Handle missing values using mean imputation
- Apply coefficient of variation to select informative features
- Visualize correlation heatmaps of features
- Train Random Forest and XGBoost classifiers with SMOTE to address class imbalance
- Evaluate model performance using ROC AUC and confusion matrix

## Key Techniques
- Feature selection via statistical measures
- SMOTE-based oversampling
- Ensemble methods (Random Forest, XGBoost)
- Model evaluation with ROC, confusion matrix, accuracy

## Libraries Used
- scikit-learn
- xgboost
- imbalanced-learn
- pandas, matplotlib, seaborn
