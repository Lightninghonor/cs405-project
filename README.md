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
- `trained_model.pkl`: trained XGBoost model bundle with threshold and metadata.
- `pred_simple.csv`: prediction output for `test_simple.csv`.
- `pred_complex.csv`: prediction output for `test_complex.csv`.
- `method_comparison.csv`: comparison of several baseline methods.
- `lgbm_xgboost_comparison.csv`: comparison of LightGBM and XGBoost methods.

## Final Method

The final model is a blended temporal ensemble:

- `0.75 * ExtraTrees + 0.25 * XGBoost`

Feature engineering includes:

- Original features `f1` to `f33`
- Missing-value indicator features
- Lag features using previous time steps: `1`, `2`, `3`, `5`, `10`
- Difference features against previous time steps: `1`, `3`, `10`
- Rolling mean and rolling standard deviation with windows: `3`, `5`, `10`

Only current and past time steps are used, so the feature construction does not leak future information.

Class imbalance is handled using XGBoost's `scale_pos_weight` and ExtraTrees'
`balanced_subsample` class weighting.

The decision threshold is selected on a chronological validation split by maximizing F1 score.

## Validation Strategy

The data is split by time order, not randomly.

Because positive anomaly labels are concentrated near the end of `train.csv`, the script uses a late forward split:

- Training: first 94% of `train.csv`
- Validation: 94% to 98% of `train.csv`

The final model is then trained on the full `train.csv` and used unchanged for both test sets.

## Validation Result

The final `Blend ET 0.75 + XGBoost temporal` model achieved:

- F1: `0.9276`
- Precision: `0.9834`
- Recall: `0.8778`
- Average Precision: `0.9511`
- Threshold: `0.033006`

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
3. Train and validate the blended ET+XGBoost model using chronological validation.
4. Retrain the final model on all training data.
5. Save `trained_model.pkl`.
6. Generate:
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

## Trained Model File

The trained model is saved as:

```text
trained_model.pkl
```

It is a Python pickle bundle containing:

- `et_model`: fitted temporal `ExtraTrees` pipeline
- `xgb_model`: fitted temporal `XGBClassifier`
- `blend_weights`: weighted average coefficients
- `threshold`: selected validation threshold
- `base_cols`: original feature column names
- `validation_metrics`: validation F1, precision, recall, AP, and prediction rate
- `final_scale_pos_weight`: class imbalance weight used for final training

The same temporal feature function in `train_predict.py` should be used before calling the loaded model.

## Method Comparison

The best validation result came from the ET+XGBoost temporal blend:

| Method | F1 | Precision | Recall | AP |
|---|---:|---:|---:|---:|
| Blend ET 0.75 + XGBoost temporal | 0.9276 | 0.9834 | 0.8778 | 0.9511 |
| ExtraTrees + temporal features | 0.8594 | 0.9234 | 0.8037 | 0.8496 |
| XGBoost + raw features | 0.8512 | 0.8659 | 0.8370 | 0.9250 |
| LightGBM + temporal features | 0.6454 | 0.6983 | 0.6000 | 0.5968 |
| LightGBM + raw features | 0.5650 | 0.8692 | 0.4185 | 0.5732 |