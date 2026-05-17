"""
compare_lgbm_xgboost.py — 模型对比与融合权重搜索脚本

功能：
  1. 在同一时序切分（前94%训练 / 94%~98%验证）上评估多个单模型：
     - ExtraTrees（时序特征）
     - LightGBM（原始特征 / 时序特征）
     - XGBoost（原始特征 / 时序特征）
  2. 对 ExtraTrees + XGBoost（时序特征）进行网格搜索融合权重（步长 0.05）
  3. 将所有结果写出到 validation_comparison.csv，按 F1 降序排列

本脚本的函数（build_temporal_features、choose_threshold、make_et_temporal_model、
make_xgb_temporal_model、positive_proba）同时被 walk_forward_eval.py 复用。
"""

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


# ── 常量 ───────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
# 主验证切分比例（与 train_predict.py 保持一致）
TRAIN_END_RATIO = 0.94
VAL_END_RATIO = 0.98
# 单模型 + 融合结果统一写入同一文件
VALIDATION_COMPARISON_CSV = "validation_comparison.csv"


def make_et_temporal_model() -> Pipeline:
    """
    构建 ExtraTrees 时序模型 Pipeline（供本脚本和 walk_forward_eval.py 共用）。

    Pipeline 步骤：
      1. SimpleImputer(strategy="median")：中位数填充 NaN
      2. ExtraTreesClassifier：
         - class_weight="balanced_subsample"：每棵树子样本内自动平衡类别权重
         - max_features="sqrt"：随机特征子集，增强集成多样性
         - min_samples_leaf=2：防止叶节点过小导致过拟合
    """
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
    """
    构建 XGBoost 时序模型（供本脚本和 walk_forward_eval.py 共用）。

    参数：
      scale_pos_weight — 正类权重 = 负样本数/正样本数，补偿类别不平衡

    关键超参：
      max_depth=3, min_child_weight=5 — 浅树防止噪声过拟合
      reg_lambda=3.0                  — L2 正则化
      eval_metric="aucpr"             — 以 PR-AUC 为内部评估目标
    """
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
    """
    构建时序衍生特征（与 train_predict.py 完全相同，供多个脚本复用）。

    特征类型（对每个原始特征列分别计算）：
      - 原始值
      - 缺失值指示（_isna）
      - Lag 特征（_lag1/2/3/5/10）：历史时刻的值
      - Diff 特征（_diff1/3/10）：与历史时刻的差值
      - 滚动均值（_rmean3/5/10）：局部趋势
      - 滚动标准差（_rstd3/5/10）：局部波动性

    所有特征仅使用当前及过去时刻，不引入未来信息。
    """
    base = df[base_cols]
    pieces = [base]

    # 缺失值指示：NaN → 1，有值 → 0
    missing = base.isna().astype(np.int8)
    missing.columns = [f"{col}_isna" for col in base_cols]
    pieces.append(missing)

    # Lag 特征：shift(k) 得到前 k 步的历史值
    for lag in (1, 2, 3, 5, 10):
        lagged = base.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in base_cols]
        pieces.append(lagged)

    # Diff 特征：当前值与前 k 步的差，反映变化速率
    for lag in (1, 3, 10):
        diffed = base - base.shift(lag)
        diffed.columns = [f"{col}_diff{lag}" for col in base_cols]
        pieces.append(diffed)

    # 滚动统计：min_periods=1 避免序列开头产生全 NaN
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
    """
    在给定标签和概率上，通过 PR 曲线选取使 F1 最大的决策阈值。

    返回包含以下键的字典：
      threshold  — 最优决策阈值
      f1         — 对应的 F1 分数
      precision  — 对应的精确率
      recall     — 对应的召回率
      ap         — PR 曲线下面积（Average Precision）
      pred_rate  — 使用该阈值时的正例预测比例
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # 加 1e-12 防止 P+R=0 时除零
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))  # 排除最后一个（阈值=1.0）
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
    """
    网格搜索两个模型的融合权重，评估每种权重组合的验证集性能。

    融合方式：blend_prob = left_w * left_prob + (1 - left_w) * right_prob

    参数：
      y_val        — 验证集真实标签
      left_prob    — 左模型（ET）的正类概率
      right_prob   — 右模型（XGBoost）的正类概率
      left_name    — 左模型名称（用于结果标注）
      right_name   — 右模型名称
      left_weights — 左模型权重候选列表（如 [0.05, 0.10, ..., 0.95]）
      feature_set  — 特征集名称（用于结果标注）

    返回：
      每种权重组合对应的指标字典列表
    """
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
    """
    从模型的 predict_proba 输出中安全提取正类（label=1）的概率。

    处理以下边界情况：
      - 输出为 1D 数组（某些模型直接返回正类概率）
      - 输出为 2 列（标准二分类，取 classes_==1 对应列）
      - 验证集中只有单一类别（全负例或全正例，返回全 0 或全 1）

    参数：
      model — 已训练的分类器（需有 predict_proba 方法）
      x     — 特征 DataFrame

    返回：
      正类概率的 1D numpy 数组
    """
    prob = model.predict_proba(x)
    if prob.ndim == 1:
        return prob
    if prob.shape[1] == 2:
        # 优先通过 classes_ 属性确定正类列索引，避免列顺序假设
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            pos_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
            return prob[:, pos_idx]
        return prob[:, 1]

    # 单类情况：模型只见过一种标签
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 1 and int(classes[0]) == 1:
        return np.ones(prob.shape[0], dtype=float)
    return np.zeros(prob.shape[0], dtype=float)


def make_models(scale_pos_weight: float) -> list[tuple[str, str, object]]:
    """
    构建所有待对比的模型列表。

    返回：
      [(模型名称, 特征集类型, 模型实例), ...]
      特征集类型为 "raw"（原始33列）或 "temporal"（时序衍生特征）

    包含的模型：
      1. ExtraTrees temporal  — 基准模型，时序特征 + balanced_subsample
      2. LightGBM raw         — LightGBM 原始特征，用于对比时序特征的增益
      3. LightGBM temporal    — LightGBM 时序特征
      4. XGBoost raw          — XGBoost 原始特征
      5. XGBoost temporal     — XGBoost 时序特征（最终融合的组件之一）
    """
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
                verbosity=-1,  # 关闭 LightGBM 训练日志
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
    # ── 1. 加载数据与时序切分 ──────────────────────────────────────────────────
    train = pd.read_csv("train.csv")
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    train_end = int(len(train) * TRAIN_END_RATIO)
    val_end = int(len(train) * VAL_END_RATIO)
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    # 正类权重：仅基于训练子集计算，防止验证集信息泄露
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))

    print(
        f"split: train_pos={int(y_train.sum())}, val_pos={int(y_val.sum())}, "
        f"scale_pos_weight={scale_pos_weight:.2f}",
        flush=True,
    )

    # ── 2. 准备两种特征集 ──────────────────────────────────────────────────────
    x_raw = train[base_cols]          # 原始 33 列特征
    print("Building temporal features...", flush=True)
    x_temporal = build_temporal_features(train, base_cols)  # 时序衍生特征（~528列）

    split_label = f"train_end={TRAIN_END_RATIO:.2f}_val_end={VAL_END_RATIO:.2f}"
    results: list[dict[str, float | str]] = []
    fitted_by_method: dict[str, object] = {}

    # ── 3. 逐一训练并评估单模型 ────────────────────────────────────────────────
    for method, feature_set, model in make_models(scale_pos_weight):
        # 根据模型类型选择对应特征集
        x = x_temporal if feature_set == "temporal" else x_raw
        start = time.time()
        print(f"Fitting {method}...", flush=True)
        model.fit(x.iloc[:train_end], y_train)
        fitted_by_method[method] = model  # 保存已训练模型，供后续融合使用

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

    # ── 4. 网格搜索 ET + XGBoost 融合权重 ─────────────────────────────────────
    # 复用上面已训练的模型，无需重新拟合
    et_name = "ExtraTrees temporal baseline"
    xgb_name = "XGBoost temporal"
    et_model = fitted_by_method[et_name]
    xgb_model = fitted_by_method[xgb_name]
    et_val_prob = positive_proba(et_model, x_temporal.iloc[train_end:val_end])
    xgb_val_prob = positive_proba(xgb_model, x_temporal.iloc[train_end:val_end])

    # 步长 0.05，覆盖 ET 权重从 0.05 到 0.95 的所有组合
    blend_weights = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
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

    # ── 5. 汇总输出 ────────────────────────────────────────────────────────────
    # 单模型 + 所有融合权重结果合并，按 F1 降序排列
    combined = pd.DataFrame(results + blend_rows).sort_values(["f1", "ap"], ascending=False)
    combined.to_csv(VALIDATION_COMPARISON_CSV, index=False)

    single_df = pd.DataFrame(results).sort_values(["f1", "ap"], ascending=False)
    blend_out = pd.DataFrame(blend_rows).sort_values(["f1", "ap"], ascending=False)

    print("\nSUMMARY (single models)")
    print(
        single_df[
            ["method", "features", "f1", "precision", "recall", "ap", "threshold", "pred_rate", "seconds"]
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
