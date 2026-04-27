from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42


def build_temporal_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
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


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    pred = (prob >= threshold).astype(np.int8)

    return {
        "threshold": threshold,
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "ap": float(average_precision_score(y_true, prob)),
        "pred_rate": float(pred.mean()),
    }


def make_models(scale_pos_weight: float) -> list[tuple[str, str, object]]:
    return [
        (
            "ExtraTrees temporal baseline",
            "temporal",
            Pipeline(
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
            ),
        ),
        (
            "LightGBM raw",
            "raw",
            LGBMClassifier(
                objective="binary",
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),
        ),
        (
            "LightGBM temporal",
            "temporal",
            LGBMClassifier(
                objective="binary",
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),
        ),
        (
            "XGBoost raw",
            "raw",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                n_estimators=450,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "XGBoost temporal",
            "temporal",
            XGBClassifier(
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
            ),
        ),
    ]


def main() -> None:
    train = pd.read_csv("train.csv")
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    train_end = int(len(train) * 0.94)
    val_end = int(len(train) * 0.98)
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))

    print(
        f"split: train_pos={int(y_train.sum())}, val_pos={int(y_val.sum())}, "
        f"scale_pos_weight={scale_pos_weight:.2f}",
        flush=True,
    )

    x_raw = train[base_cols]
    print("Building temporal features...", flush=True)
    x_temporal = build_temporal_features(train, base_cols)

    results = []
    for method, feature_set, model in make_models(scale_pos_weight):
        x = x_temporal if feature_set == "temporal" else x_raw
        start = time.time()
        print(f"Fitting {method}...", flush=True)
        model.fit(x.iloc[:train_end], y_train)
        prob = model.predict_proba(x.iloc[train_end:val_end])[:, 1]
        metrics = choose_threshold(y_val, prob)
        metrics.update(
            {
                "method": method,
                "features": feature_set,
                "seconds": round(time.time() - start, 1),
            }
        )
        results.append(metrics)
        print(metrics, flush=True)

    out = pd.DataFrame(results).sort_values(["f1", "ap"], ascending=False)
    out.to_csv("lgbm_xgboost_comparison.csv", index=False)
    print("\nSUMMARY")
    print(
        out[
            [
                "method",
                "features",
                "f1",
                "precision",
                "recall",
                "ap",
                "threshold",
                "pred_rate",
                "seconds",
            ]
        ].to_string(index=False)
    )
    print("\nWrote lgbm_xgboost_comparison.csv")


if __name__ == "__main__":
    main()
