# CS405 Project: Robust Time-Series Anomaly Detection

This project builds a supervised machine learning model for anomaly detection in noisy and imbalanced time-series data.

The final submission uses one trained model to generate predictions for both:

- `test_simple.csv`
- `test_complex.csv`

No test labels or external labeled data are used.

## Files

- `train.csv`: labeled training data. The target column is `y`.
- `test_simple.csv`: unlabeled test data for Task 1.
- `test_complex.csv`: unlabeled test data for Task 2.
- `train_predict.py`: final training and prediction script.
- `pred_simple.csv`: prediction output for `test_simple.csv`.
- `pred_complex.csv`: prediction output for `test_complex.csv`.
- `method_comparison.csv`: comparison of several baseline methods.
- `lgbm_xgboost_comparison.csv`: comparison of LightGBM and XGBoost methods.

## Final Method

The final model is `XGBoost` with temporal feature engineering.

Feature engineering includes:

- Original features `f1` to `f33`
- Missing-value indicator features
- Lag features using previous time steps: `1`, `2`, `3`, `5`, `10`
- Difference features against previous time steps: `1`, `3`, `10`
- Rolling mean and rolling standard deviation with windows: `3`, `5`, `10`

Only current and past time steps are used, so the feature construction does not leak future information.

Class imbalance is handled using XGBoost's `scale_pos_weight`.

The decision threshold is selected on a chronological validation split by maximizing F1 score.

## Validation Strategy

The data is split by time order, not randomly.

Because positive anomaly labels are concentrated near the end of `train.csv`, the script uses a late forward split:

- Training: first 94% of `train.csv`
- Validation: 94% to 98% of `train.csv`

The final model is then trained on the full `train.csv` and used unchanged for both test sets.

## Validation Result

The final `XGBoost + temporal features` model achieved:

- F1: `0.8944`
- Precision: `0.9283`
- Recall: `0.8630`
- Average Precision: `0.9490`
- Threshold: `0.122445`

## Environment

A virtual environment was created for LightGBM and XGBoost experiments:

```bash
python -m venv --system-site-packages .venv_lgbm_xgb
.venv_lgbm_xgb/bin/python -m pip install --upgrade pip
.venv_lgbm_xgb/bin/python -m pip install lightgbm xgboost
```

The final script should be run with this environment:

```bash
.venv_lgbm_xgb/bin/python train_predict.py
```

## Reproducing Predictions

Run:

```bash
.venv_lgbm_xgb/bin/python train_predict.py
```

This will:

1. Load `train.csv`, `test_simple.csv`, and `test_complex.csv`.
2. Build temporal features.
3. Train and validate the XGBoost model using chronological validation.
4. Retrain the final model on all training data.
5. Generate:
   - `pred_simple.csv`
   - `pred_complex.csv`

## Output Format

Both prediction files follow the required format:

```csv
y_pred
0
1
0
```

The generated files have been checked:

- `pred_simple.csv`: 25647 rows, 953 predicted anomalies
- `pred_complex.csv`: 34542 rows, 831 predicted anomalies

## Method Comparison

The best validation result came from XGBoost with temporal features:

| Method | F1 | Precision | Recall | AP |
|---|---:|---:|---:|---:|
| XGBoost + temporal features | 0.8944 | 0.9283 | 0.8630 | 0.9490 |
| ExtraTrees + temporal features | 0.8594 | 0.9234 | 0.8037 | 0.8496 |
| XGBoost + raw features | 0.8512 | 0.8659 | 0.8370 | 0.9250 |
| LightGBM + temporal features | 0.6454 | 0.6983 | 0.6000 | 0.5968 |
| LightGBM + raw features | 0.5650 | 0.8692 | 0.4185 | 0.5732 |