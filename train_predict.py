from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42
TRAIN_PATH = Path("train.csv")
TEST_SIMPLE_PATH = Path("test_simple.csv")
TEST_COMPLEX_PATH = Path("test_complex.csv")
PRED_SIMPLE_PATH = Path("pred_simple.csv")
PRED_COMPLEX_PATH = Path("pred_complex.csv")
MODEL_PATH = Path("trained_model.pkl")


def build_temporal_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """Create current and past-only features, so validation remains chronological."""
    base = df[base_cols]
    pieces = [base]

    missing = base.isna().astype(np.int8)
    missing.columns = [f"{col}_isna" for col in base_cols]
    pieces.append(missing)

    for lag in (1, 2, 3, 5, 10):
        lagged = base.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in base_cols]
        pieces.append(lagged)

    for lag in (1, 3, 10):
        diffed = base - base.shift(lag)
        diffed.columns = [f"{col}_diff{lag}" for col in base_cols]
        pieces.append(diffed)

    for window in (3, 5, 10):
        rolling = base.rolling(window=window, min_periods=1)

        mean = rolling.mean()
        mean.columns = [f"{col}_rmean{window}" for col in base_cols]
        pieces.append(mean)

        std = rolling.std()
        std.columns = [f"{col}_rstd{window}" for col in base_cols]
        pieces.append(std)

    return pd.concat(pieces, axis=1)


def make_xgb_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=350,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_et_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    if len(thresholds) == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ap": 0.0}

    best_idx = int(np.nanargmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    metrics = {
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "ap": float(average_precision_score(y_true, prob)),
    }
    return threshold, metrics


def write_predictions(path: Path, prob: np.ndarray, threshold: float) -> None:
    pred = (prob >= threshold).astype(np.int8)
    pd.DataFrame({"y_pred": pred}).to_csv(path, index=False)
    print(f"Wrote {path}: {len(pred)} rows, predicted positives={int(pred.sum())}", flush=True)


def save_model_bundle(
    path: Path,
    xgb_model: XGBClassifier,
    et_model: Pipeline,
    threshold: float,
    base_cols: list[str],
    metrics: dict[str, float],
    final_scale_pos_weight: float,
) -> None:
    bundle = {
        "xgb_model": xgb_model,
        "et_model": et_model,
        "blend_weights": {"et": 0.75, "xgb": 0.25},
        "threshold": threshold,
        "base_cols": base_cols,
        "validation_metrics": metrics,
        "final_scale_pos_weight": final_scale_pos_weight,
        "feature_builder": "build_temporal_features in train_predict.py",
    }
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test_simple = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex = pd.read_csv(TEST_COMPLEX_PATH)

    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    print("Building temporal features...", flush=True)
    x_train = build_temporal_features(train, base_cols)
    x_simple = build_temporal_features(test_simple, base_cols)
    x_complex = build_temporal_features(test_complex, base_cols)

    # Positives are concentrated near the end of train.csv, so this late forward
    # split keeps chronological order while still giving train/validation positives.
    train_end = int(len(train) * 0.94)
    val_end = int(len(train) * 0.98)
    val_scale_pos_weight = float(
        (train_end - y[:train_end].sum()) / max(1, y[:train_end].sum())
    )

    val_et_model = make_et_model()
    val_xgb_model = make_xgb_model(val_scale_pos_weight)
    print("Fitting validation blend models (ET + XGBoost)...", flush=True)
    val_et_model.fit(x_train.iloc[:train_end], y[:train_end])
    val_xgb_model.fit(x_train.iloc[:train_end], y[:train_end])
    val_et_prob = val_et_model.predict_proba(x_train.iloc[train_end:val_end])[:, 1]
    val_xgb_prob = val_xgb_model.predict_proba(x_train.iloc[train_end:val_end])[:, 1]
    val_prob = 0.75 * val_et_prob + 0.25 * val_xgb_prob
    threshold, metrics = choose_threshold(y[train_end:val_end], val_prob)

    val_pred = (val_prob >= threshold).astype(np.int8)
    metrics["predicted_positive_rate"] = float(val_pred.mean())
    print(
        "Validation:",
        f"threshold={threshold:.6f}",
        f"f1={metrics['f1']:.4f}",
        f"precision={metrics['precision']:.4f}",
        f"recall={metrics['recall']:.4f}",
        f"ap={metrics['ap']:.4f}",
        f"pred_rate={metrics['predicted_positive_rate']:.4f}",
        flush=True,
    )

    final_scale_pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))
    final_et_model = make_et_model()
    final_xgb_model = make_xgb_model(final_scale_pos_weight)
    print("Fitting final blend models on all training data...", flush=True)
    final_et_model.fit(x_train, y)
    final_xgb_model.fit(x_train, y)

    print("Predicting test files...", flush=True)
    simple_prob = (
        0.75 * final_et_model.predict_proba(x_simple)[:, 1]
        + 0.25 * final_xgb_model.predict_proba(x_simple)[:, 1]
    )
    complex_prob = (
        0.75 * final_et_model.predict_proba(x_complex)[:, 1]
        + 0.25 * final_xgb_model.predict_proba(x_complex)[:, 1]
    )

    write_predictions(PRED_SIMPLE_PATH, simple_prob, threshold)
    write_predictions(PRED_COMPLEX_PATH, complex_prob, threshold)
    save_model_bundle(
        MODEL_PATH,
        final_xgb_model,
        final_et_model,
        threshold,
        base_cols,
        metrics,
        final_scale_pos_weight,
    )


if __name__ == "__main__":
    main()
