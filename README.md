# Advanced ML & Statistical Modeling

Independent synthetic-data portfolio project demonstrating **XGBoost**, **Optuna** hyperparameter optimization, **SHAP** explainability, bootstrap confidence intervals, a paired permutation hypothesis test, and **MLflow** tracking.

Run: `pip install -r requirements.txt` then `python train.py`.

The script uses a stratified untouched holdout set, three-fold CV on development data, a logistic-regression baseline, PR-AUC/ROC-AUC, and optional SHAP explanations. Data and results are synthetic; no real-world performance is claimed. Run tests with `python -m pytest`.
