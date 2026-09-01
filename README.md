# Credit Scoring: Model Selection and Optimization

Classification model estimating the probability that a client repays a loan, then
labelling the application as granted or refused.

## Approach

Model comparison by cross-validation, hyperparameter tuning with Optuna, feature
importance analysis, business-cost threshold optimization, MLflow model registry
and serving.

## Result

Final model: optimized LightGBM. Business cost 29778.2 (+/- 546.9) on 5-fold
cross-validation, mean AUC 0.79, optimal business threshold 0.48.

## Stack

scikit-learn, LightGBM, XGBoost, MLflow, Optuna, SHAP, pandas, NumPy, matplotlib,
seaborn, plotly, joblib

## Contents

- `projet6.ipynb`: exploration and model comparison
- `optimisations.ipynb`: hyperparameter tuning and threshold optimization
- `export_champion.ipynb`: champion model export
- `model.joblib`: champion model, deployed as an API in P08

Input data and MLflow artifacts are excluded from this repository for size reasons.

## Related

The model in this repository is the one served by
[P08 Credit Scoring MLOps](https://github.com/Jojo4911/pret-a-depenser), which
covers deployment, CI/CD and drift monitoring.
