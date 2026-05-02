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
# 主验证切分（单模型与 ET+XGB 融合共用，保证参数与数据一致）
TRAIN_END_RATIO = 0.94
VAL_END_RATIO = 0.98
# 单模型 + 融合主结果输出到同一文件
VALIDATION_COMPARISON_CSV = "validation_comparison.csv"


def make_et_temporal_model() -> Pipeline:
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


def make_xgb_temporal_model(scale_pos_weight: float) -> XGBClassifier:
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


def evaluate_blend_grid(
    y_val: np.ndarray,
    left_prob: np.ndarray,
    right_prob: np.ndarray,
    left_name: str,
    right_name: str,
    left_weights: list[float],
    feature_set: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for left_w in left_weights:
        right_w = 1.0 - left_w
        blend_prob = left_w * left_prob + right_w * right_prob
        metrics = choose_threshold(y_val, blend_prob)
        metrics["method"] = f"Blend {left_name} {left_w:.2f} + {right_name} {right_w:.2f} ({feature_set})"
        metrics["seconds"] = 0.0
        rows.append(metrics)
    return rows


def positive_proba(model: object, x: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(x)
    if prob.ndim == 1:
        return prob
    if prob.shape[1] == 2:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            pos_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
            return prob[:, pos_idx]
        return prob[:, 1]

    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
        return np.ones(prob.shape[0], dtype=float)
    return np.zeros(prob.shape[0], dtype=float)


def make_models(scale_pos_weight: float) -> list[tuple[str, str, object]]:
    return [
        (
            "ExtraTrees temporal baseline",
            "temporal",
            make_et_temporal_model(),
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
            make_xgb_temporal_model(scale_pos_weight),
        ),
    ]


def main() -> None:
    train = pd.read_csv("train.csv")
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    train_end = int(len(train) * TRAIN_END_RATIO)
    val_end = int(len(train) * VAL_END_RATIO)
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

    split_label = f"train_end={TRAIN_END_RATIO:.2f}_val_end={VAL_END_RATIO:.2f}"
    results: list[dict[str, float | str]] = []
    fitted_by_method: dict[str, object] = {}

    for method, feature_set, model in make_models(scale_pos_weight):
        x = x_temporal if feature_set == "temporal" else x_raw
        start = time.time()
        print(f"Fitting {method}...", flush=True)
        model.fit(x.iloc[:train_end], y_train)
        fitted_by_method[method] = model
        prob = positive_proba(model, x.iloc[train_end:val_end])
        metrics = choose_threshold(y_val, prob)
        metrics.update(
            {
                "eval_type": "single_model",
                "method": method,
                "features": feature_set,
                "split": split_label,
                "seconds": round(time.time() - start, 1),
            }
        )
        results.append(metrics)
        print(metrics, flush=True)

    et_name = "ExtraTrees temporal baseline"
    xgb_name = "XGBoost temporal"
    et_model = fitted_by_method[et_name]
    xgb_model = fitted_by_method[xgb_name]
    et_val_prob = positive_proba(et_model, x_temporal.iloc[train_end:val_end])
    xgb_val_prob = positive_proba(xgb_model, x_temporal.iloc[train_end:val_end])

    blend_weights = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]
    blend_rows = evaluate_blend_grid(
        y_val=y_val,
        left_prob=et_val_prob,
        right_prob=xgb_val_prob,
        left_name="ET",
        right_name="XGBoost",
        left_weights=blend_weights,
        feature_set="temporal",
    )
    for row in blend_rows:
        row["eval_type"] = "et_xgb_blend"
        row["features"] = "temporal"
        row["split"] = split_label

    combined = pd.DataFrame(results + blend_rows).sort_values(["f1", "ap"], ascending=False)
    combined.to_csv(VALIDATION_COMPARISON_CSV, index=False)
    single_df = pd.DataFrame(results).sort_values(["f1", "ap"], ascending=False)
    blend_out = pd.DataFrame(blend_rows).sort_values(["f1", "ap"], ascending=False)

    print("\nSUMMARY (single models)")
    print(
        single_df[
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
    print("\nBLEND SUMMARY (same ET/XGB fits as single models above)")
    print(
        blend_out[
            ["method", "f1", "precision", "recall", "ap", "threshold", "pred_rate", "seconds"]
        ].to_string(index=False)
    )
    print(f"\nWrote {VALIDATION_COMPARISON_CSV} (single + blend)")


if __name__ == "__main__":
    main()
