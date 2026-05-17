"""
experiment_new.py — 新算法与特征工程实验脚本
==============================================

在保留原有 ET+XGBoost 融合方案的基础上，探索以下新方向：

【新增方向 A】扩展时序特征工程
  - 更长 lag（15, 20）和更大滚动窗口（15, 20）
  - 滚动最大值/最小值/极差（捕捉局部极端值）
  - 指数加权移动平均（EWMA，对近期更敏感）
  - 跨特征交叉统计（行级均值/标准差/最大值，捕捉多传感器协同异常）

【新增方向 B】LightGBM DART（Dropout Additive Regression Trees）
  - DART boosting 通过随机丢弃树来防止过拟合，对噪声时序更鲁棒

【新增方向 C】HistGradientBoosting（sklearn 原生，支持原生缺失值）
  - 无需 Imputer，直接处理 NaN，对 test_complex 的分布偏移更鲁棒

【新增方向 D】Isolation Forest 异常分数作为元特征
  - 无监督异常检测分数作为额外特征，增强对 test_complex 未知分布的泛化

【新增方向 E】Stacking 二层融合
  - 第一层：ET + XGBoost + LightGBM DART + HistGB
  - 第二层：Logistic Regression 学习最优融合权重（替代手动网格搜索）

【新增方向 F】阈值优化改进
  - 在 F1 基础上额外考虑 F-beta（beta=0.5，更重视精确率）
  - 对 Task2 使用更保守阈值（提高精确率，降低误报）

所有方向均在同一时序切分（前95%训练 / 95%~99%验证）上评估，
结果写入 experiment_comparison.csv，按 F1 降序排列。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
# 【新增 MCC】引入 matthews_corrcoef
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── 常量 ───────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_END_RATIO = 0.95
VAL_END_RATIO = 0.99
EXPERIMENT_CSV = "experiment_comparison.csv"
PRED_SIMPLE_NEW = "pred_simple_new.csv"
PRED_COMPLEX_NEW = "pred_complex_new.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 【原有】基础时序特征（与 train_predict.py 完全一致，保留不变）
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_features_original(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """原有时序特征构建（保留原始实现，不做修改）。"""
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


# ══════════════════════════════════════════════════════════════════════════════
# 【新增方向 A】扩展时序特征工程
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_features_extended(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    【新增 A】在原有时序特征基础上扩展：
      - 更长 lag（15, 20）：捕捉更长周期的历史依赖
      - 更大滚动窗口（15, 20）：捕捉更长期趋势
      - 滚动最大值/最小值/极差：捕捉局部极端值，异常往往表现为极端偏离
      - EWMA（span=5, 10）：指数加权均值，对近期变化更敏感
      - 行级跨特征统计：多传感器协同异常检测
    """
    base = df[base_cols]
    pieces = [base]

    # ── 原有特征（保留）──
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

    # ── 【新增 A1】更长 lag 特征 ──
    for lag in (15, 20):
        lagged = base.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in base_cols]
        pieces.append(lagged)

    # ── 【新增 A2】更大滚动窗口的均值和标准差 ──
    for window in (15, 20):
        rolling = base.rolling(window=window, min_periods=1)
        mean = rolling.mean()
        mean.columns = [f"{col}_rmean{window}" for col in base_cols]
        pieces.append(mean)
        std = rolling.std()
        std.columns = [f"{col}_rstd{window}" for col in base_cols]
        pieces.append(std)

    # ── 【新增 A3】滚动最大值、最小值、极差（max-min）──
    # 异常点往往在局部窗口内产生极端值，极差特征能直接捕捉这种模式
    for window in (5, 10):
        rolling = base.rolling(window=window, min_periods=1)
        rmax = rolling.max()
        rmax.columns = [f"{col}_rmax{window}" for col in base_cols]
        pieces.append(rmax)
        rmin = rolling.min()
        rmin.columns = [f"{col}_rmin{window}" for col in base_cols]
        pieces.append(rmin)
        rrange = rmax.values - rmin.values
        rrange_df = pd.DataFrame(rrange, columns=[f"{col}_rrange{window}" for col in base_cols],
                                 index=base.index)
        pieces.append(rrange_df)

    # ── 【新增 A4】指数加权移动平均（EWMA）──
    # EWMA 对近期数据赋予更高权重，比简单滚动均值更能捕捉突变
    for span in (5, 10):
        ewm = base.ewm(span=span, min_periods=1).mean()
        ewm.columns = [f"{col}_ewma{span}" for col in base_cols]
        pieces.append(ewm)
        # EWMA 残差：当前值与 EWMA 的偏差，直接反映异常程度
        ewm_resid = base.values - ewm.values
        ewm_resid_df = pd.DataFrame(
            ewm_resid,
            columns=[f"{col}_ewma_resid{span}" for col in base_cols],
            index=base.index,
        )
        pieces.append(ewm_resid_df)

    # ── 【新增 A5】行级跨特征统计（多传感器协同异常）──
    # 当多个传感器同时异常时，行级统计量会显著偏离正常范围
    row_mean = base.mean(axis=1).rename("row_mean")
    row_std = base.std(axis=1).rename("row_std")
    row_max = base.max(axis=1).rename("row_max")
    row_min = base.min(axis=1).rename("row_min")
    row_range = (row_max - row_min).rename("row_range")
    # 行级缺失比例：test_complex 中缺失模式可能与 train 不同
    row_missing_rate = base.isna().mean(axis=1).rename("row_missing_rate")
    pieces.extend([row_mean, row_std, row_max, row_min, row_range, row_missing_rate])

    return pd.concat(pieces, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 【新增方向 D】Isolation Forest 异常分数元特征
# ══════════════════════════════════════════════════════════════════════════════
def add_isolation_forest_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_simple: pd.DataFrame,
    x_complex: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    【新增 D】用 Isolation Forest 在训练集上拟合，
    为所有数据集生成无监督异常分数作为额外特征。

    原理：
      - Isolation Forest 通过随机分割隔离样本，异常点需要更少的分割步骤
      - 其 decision_function 输出越低表示越异常
      - 作为元特征加入监督模型，可以提供无监督视角的异常信号
      - 对 test_complex 的分布偏移有一定鲁棒性（无监督，不依赖标签分布）

    注意：仅在训练集上 fit，避免信息泄露。
    """
    # 用中位数填充 NaN（Isolation Forest 不支持缺失值）
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_val_imp = imputer.transform(x_val)
    x_simple_imp = imputer.transform(x_simple)
    x_complex_imp = imputer.transform(x_complex)

    # 训练 Isolation Forest（contamination 设为训练集正例比例的估计）
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # 估计异常比例约 5%
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(x_train_imp)

    # decision_function 输出：越低越异常（负值表示异常）
    def add_iso_col(x_imp: np.ndarray, x_df: pd.DataFrame) -> pd.DataFrame:
        score = iso.decision_function(x_imp)
        return x_df.assign(iso_score=score)

    return (
        add_iso_col(x_train_imp, x_train),
        add_iso_col(x_val_imp, x_val),
        add_iso_col(x_simple_imp, x_simple),
        add_iso_col(x_complex_imp, x_complex),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 阈值选择工具函数
# ══════════════════════════════════════════════════════════════════════════════
def choose_threshold_f1(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict]:
    """原有 F1 最大化阈值选择（保留原始逻辑）。"""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ap": 0.0, "mcc": 0.0}
    best_idx = int(np.nanargmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    # 【新增 MCC】在最优 F1 阈值下计算 MCC，衡量综合分类质量
    pred_at_best = (prob >= threshold).astype(np.int8)
    mcc = float(matthews_corrcoef(y_true, pred_at_best))
    metrics = {
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "ap": float(average_precision_score(y_true, prob)),
        "mcc": mcc,       # 【新增 MCC】Matthews 相关系数，-1~1，不受类别不平衡影响
        "threshold": threshold,
    }
    return threshold, metrics


def choose_threshold_fbeta(
    y_true: np.ndarray, prob: np.ndarray, beta: float = 0.5
) -> tuple[float, dict]:
    """
    【新增 F】F-beta 阈值选择。
    beta < 1 时更重视精确率（减少误报），适合 Task1 高精确率场景。
    beta > 1 时更重视召回率（减少漏报），适合 Task2 鲁棒性场景。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ap": 0.0, "mcc": 0.0}
    best_idx = int(np.nanargmax(fbeta[:-1]))
    threshold = float(thresholds[best_idx])
    f1_at_best = float(
        2 * precision[best_idx] * recall[best_idx]
        / (precision[best_idx] + recall[best_idx] + 1e-12)
    )
    # 【新增 MCC】在 F-beta 最优阈值下同样计算 MCC
    pred_at_best = (prob >= threshold).astype(np.int8)
    mcc = float(matthews_corrcoef(y_true, pred_at_best))
    metrics = {
        "f1": f1_at_best,
        "fbeta": float(fbeta[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "ap": float(average_precision_score(y_true, prob)),
        "mcc": mcc,       # 【新增 MCC】
        "threshold": threshold,
    }
    return threshold, metrics


def print_metrics(name: str, metrics: dict, elapsed: float = 0.0) -> None:
    """格式化打印指标。"""
    print(
        f"[{name}] "
        f"F1={metrics.get('f1', 0):.4f} "
        f"P={metrics.get('precision', 0):.4f} "
        f"R={metrics.get('recall', 0):.4f} "
        f"AP={metrics.get('ap', 0):.4f} "
        f"MCC={metrics.get('mcc', 0):.4f} "  # 【新增 MCC】
        f"thr={metrics.get('threshold', 0):.6f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 模型构建函数
# ══════════════════════════════════════════════════════════════════════════════
def make_et_model() -> Pipeline:
    """原有 ExtraTrees Pipeline（保留不变）。"""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=250, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def make_xgb_model(scale_pos_weight: float) -> XGBClassifier:
    """原有 XGBoost 模型（保留不变）。"""
    return XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_estimators=350, learning_rate=0.03, max_depth=3, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=RANDOM_STATE,
    )


def make_lgbm_dart_model(scale_pos_weight: float) -> LGBMClassifier:
    """
    【新增 B】LightGBM DART boosting。
    DART 在每次迭代时随机丢弃已有的树，类似 Dropout，
    能有效防止过拟合，对噪声时序数据更鲁棒。
    """
    return LGBMClassifier(
        boosting_type="dart",       # DART：Dropout Additive Regression Trees
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        drop_rate=0.1,              # 每次迭代丢弃 10% 的树
        skip_drop=0.5,              # 50% 概率跳过 dropout（保持稳定性）
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def make_histgb_model() -> HistGradientBoostingClassifier:
    """
    【新增 C】sklearn HistGradientBoosting。
    原生支持 NaN（无需 Imputer），对 test_complex 的缺失模式变化更鲁棒。
    class_weight="balanced" 自动处理类别不平衡。
    """
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",    # 自动平衡类别权重
        random_state=RANDOM_STATE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 【新增方向 E】Stacking 二层融合
# ══════════════════════════════════════════════════════════════════════════════
def fit_stacking_meta_learner(
    prob_matrix_train: np.ndarray,
    y_train: np.ndarray,
    prob_matrix_val: np.ndarray,
) -> np.ndarray:
    """
    【新增 E】用 Logistic Regression 作为 Stacking 元学习器。

    原理：
      第一层各模型在训练集上生成概率（需用交叉验证避免泄露，
      此处简化为直接用验证集概率学习权重）。
      元学习器自动学习最优融合权重，比手动网格搜索更精确。

    参数：
      prob_matrix_train — 形状 (n_train, n_models)，第一层在训练集上的概率
      y_train           — 训练集标签
      prob_matrix_val   — 形状 (n_val, n_models)，第一层在验证集上的概率

    返回：
      meta_prob — 元学习器在验证集上的融合概率
    """
    # 使用 Logistic Regression 学习模型权重，C=1.0 适度正则化
    meta = LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)
    meta.fit(prob_matrix_train, y_train)
    meta_prob = meta.predict_proba(prob_matrix_val)[:, 1]
    print(f"  Stacking meta weights (LR coef): {meta.coef_[0]}", flush=True)
    return meta_prob, meta


def main() -> None:
    print("=" * 70)
    print("实验脚本：多方向新算法对比")
    print("=" * 70)

    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    print("\n[Step 1] 加载数据...", flush=True)
    train = pd.read_csv("train.csv")
    test_simple = pd.read_csv("test_simple.csv")
    test_complex = pd.read_csv("test_complex.csv")

    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    train_end = int(len(train) * TRAIN_END_RATIO)
    val_end = int(len(train) * VAL_END_RATIO)
    y_train = y[:train_end]
    y_val = y[train_end:val_end]

    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    print(
        f"训练集大小: {train_end}, 验证集大小: {val_end - train_end}, "
        f"训练集正例: {int(y_train.sum())}, 验证集正例: {int(y_val.sum())}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}",
        flush=True,
    )

    # ── 2. 构建两种特征集 ──────────────────────────────────────────────────────
    print("\n[Step 2] 构建特征...", flush=True)

    print("  构建原始时序特征（原有方案）...", flush=True)
    x_orig = build_temporal_features_original(train, base_cols)
    x_simple_orig = build_temporal_features_original(test_simple, base_cols)
    x_complex_orig = build_temporal_features_original(test_complex, base_cols)

    print("  【新增 A】构建扩展时序特征...", flush=True)
    x_ext = build_temporal_features_extended(train, base_cols)
    x_simple_ext = build_temporal_features_extended(test_simple, base_cols)
    x_complex_ext = build_temporal_features_extended(test_complex, base_cols)

    print(
        f"  原始特征维度: {x_orig.shape[1]}, 扩展特征维度: {x_ext.shape[1]}",
        flush=True,
    )

    results = []  # 存储所有实验结果

    # ══════════════════════════════════════════════════════════════════════════
    # 【基线】原有方案：ET 0.75 + XGBoost 0.25（原始时序特征）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[基线] 原有方案: ET 0.75 + XGBoost 0.25（原始时序特征）", flush=True)
    t0 = time.time()
    baseline_et = make_et_model()
    baseline_xgb = make_xgb_model(scale_pos_weight)
    baseline_et.fit(x_orig.iloc[:train_end], y_train)
    baseline_xgb.fit(x_orig.iloc[:train_end], y_train)
    baseline_et_prob = baseline_et.predict_proba(x_orig.iloc[train_end:val_end])[:, 1]
    baseline_xgb_prob = baseline_xgb.predict_proba(x_orig.iloc[train_end:val_end])[:, 1]
    baseline_prob = 0.75 * baseline_et_prob + 0.25 * baseline_xgb_prob
    thr_bl, metrics_bl = choose_threshold_f1(y_val, baseline_prob)
    elapsed = time.time() - t0
    print_metrics("基线 ET0.75+XGB0.25 原始时序特征", metrics_bl, elapsed)
    results.append({"method": "【基线】ET 0.75 + XGBoost 0.25（原始时序特征）", **metrics_bl, "seconds": round(elapsed, 1)})

    # ── 保存基线模型用于 Stacking ──
    baseline_et_train_prob = baseline_et.predict_proba(x_orig.iloc[:train_end])[:, 1]
    baseline_xgb_train_prob = baseline_xgb.predict_proba(x_orig.iloc[:train_end])[:, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 A】扩展时序特征 + 原有 ET + XGBoost
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 A] 扩展时序特征 + ET 0.75 + XGBoost 0.25", flush=True)
    t0 = time.time()
    ext_et = make_et_model()
    ext_xgb = make_xgb_model(scale_pos_weight)
    ext_et.fit(x_ext.iloc[:train_end], y_train)
    ext_xgb.fit(x_ext.iloc[:train_end], y_train)
    ext_et_prob = ext_et.predict_proba(x_ext.iloc[train_end:val_end])[:, 1]
    ext_xgb_prob = ext_xgb.predict_proba(x_ext.iloc[train_end:val_end])[:, 1]
    ext_blend_prob = 0.75 * ext_et_prob + 0.25 * ext_xgb_prob
    thr_A, metrics_A = choose_threshold_f1(y_val, ext_blend_prob)
    elapsed = time.time() - t0
    print_metrics("方向A ET0.75+XGB0.25 扩展特征", metrics_A, elapsed)
    results.append({"method": "【方向A】扩展时序特征 ET0.75+XGB0.25", **metrics_A, "seconds": round(elapsed, 1)})

    ext_et_train_prob = ext_et.predict_proba(x_ext.iloc[:train_end])[:, 1]
    ext_xgb_train_prob = ext_xgb.predict_proba(x_ext.iloc[:train_end])[:, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 B】LightGBM DART + 扩展时序特征
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 B] LightGBM DART（扩展时序特征）", flush=True)
    t0 = time.time()
    dart_model = make_lgbm_dart_model(scale_pos_weight)
    dart_model.fit(x_ext.iloc[:train_end].fillna(x_ext.iloc[:train_end].median()), y_train)
    dart_val_prob = dart_model.predict_proba(
        x_ext.iloc[train_end:val_end].fillna(x_ext.iloc[:train_end].median())
    )[:, 1]
    thr_B, metrics_B = choose_threshold_f1(y_val, dart_val_prob)
    elapsed = time.time() - t0
    print_metrics("方向B LGBM DART 扩展特征", metrics_B, elapsed)
    results.append({"method": "【方向B】LightGBM DART（扩展时序特征）", **metrics_B, "seconds": round(elapsed, 1)})

    dart_train_prob = dart_model.predict_proba(
        x_ext.iloc[:train_end].fillna(x_ext.iloc[:train_end].median())
    )[:, 1]
    train_medians = x_ext.iloc[:train_end].median()  # 保存用于测试集填充

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 C】HistGradientBoosting（原生支持 NaN）+ 扩展时序特征
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 C] HistGradientBoosting（扩展时序特征，原生 NaN 支持）", flush=True)
    t0 = time.time()
    histgb_model = make_histgb_model()
    histgb_model.fit(x_ext.iloc[:train_end], y_train)
    histgb_val_prob = histgb_model.predict_proba(x_ext.iloc[train_end:val_end])[:, 1]
    thr_C, metrics_C = choose_threshold_f1(y_val, histgb_val_prob)
    elapsed = time.time() - t0
    print_metrics("方向C HistGB 扩展特征", metrics_C, elapsed)
    results.append({"method": "【方向C】HistGradientBoosting（扩展时序特征）", **metrics_C, "seconds": round(elapsed, 1)})

    histgb_train_prob = histgb_model.predict_proba(x_ext.iloc[:train_end])[:, 1]

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 A+B 融合】扩展特征 ET + DART（替换 XGBoost）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 A+B 融合] ET 0.6 + LGBM DART 0.4（扩展特征）", flush=True)
    t0 = time.time()
    blend_ab_prob = 0.6 * ext_et_prob + 0.4 * dart_val_prob
    thr_AB, metrics_AB = choose_threshold_f1(y_val, blend_ab_prob)
    elapsed = time.time() - t0
    print_metrics("方向AB ET0.6+DART0.4 扩展特征", metrics_AB, elapsed)
    results.append({"method": "【方向A+B】ET 0.6 + LGBM DART 0.4（扩展特征）", **metrics_AB, "seconds": round(elapsed, 1)})

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 A+C 融合】扩展特征 ET + HistGB
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 A+C 融合] ET 0.6 + HistGB 0.4（扩展特征）", flush=True)
    blend_ac_prob = 0.6 * ext_et_prob + 0.4 * histgb_val_prob
    thr_AC, metrics_AC = choose_threshold_f1(y_val, blend_ac_prob)
    print_metrics("方向AC ET0.6+HistGB0.4 扩展特征", metrics_AC)
    results.append({"method": "【方向A+C】ET 0.6 + HistGB 0.4（扩展特征）", **metrics_AC, "seconds": 0.0})

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 D】加入 Isolation Forest 元特征
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 D] Isolation Forest 元特征 + ET + XGBoost（扩展特征）", flush=True)
    t0 = time.time()
    (x_ext_tr_iso, x_ext_val_iso,
     x_simple_iso, x_complex_iso) = add_isolation_forest_features(
        x_ext.iloc[:train_end],
        x_ext.iloc[train_end:val_end],
        x_simple_ext,
        x_complex_ext,
    )
    iso_et = make_et_model()
    iso_xgb = make_xgb_model(scale_pos_weight)
    iso_et.fit(x_ext_tr_iso, y_train)
    iso_xgb.fit(x_ext_tr_iso, y_train)
    iso_et_prob = iso_et.predict_proba(x_ext_val_iso)[:, 1]
    iso_xgb_prob = iso_xgb.predict_proba(x_ext_val_iso)[:, 1]
    iso_blend_prob = 0.75 * iso_et_prob + 0.25 * iso_xgb_prob
    thr_D, metrics_D = choose_threshold_f1(y_val, iso_blend_prob)
    elapsed = time.time() - t0
    print_metrics("方向D ISO元特征+ET0.75+XGB0.25", metrics_D, elapsed)
    results.append({"method": "【方向D】IsoForest元特征+ET0.75+XGB0.25（扩展特征）", **metrics_D, "seconds": round(elapsed, 1)})

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 E】4 模型 Stacking
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 E] 4 模型 Stacking（LR 元学习器）", flush=True)
    t0 = time.time()
    # 训练集概率矩阵（第一层模型在训练集上的输出，注意这里有信息泄露风险，
    # 实际生产应用 k-fold OOF，此处为实验对比简化处理）
    stack_train_prob_matrix = np.column_stack([
        baseline_et_train_prob,   # 基线 ET（原始时序特征）
        baseline_xgb_train_prob,  # 基线 XGBoost（原始时序特征）
        ext_et_train_prob,        # 扩展特征 ET
        histgb_train_prob,        # HistGB
    ])
    stack_val_prob_matrix = np.column_stack([
        baseline_et_prob,    # 基线 ET 验证集概率
        baseline_xgb_prob,   # 基线 XGBoost 验证集概率
        ext_et_prob,         # 扩展特征 ET 验证集概率
        histgb_val_prob,     # HistGB 验证集概率
    ])
    stack_meta_prob, stack_meta_model = fit_stacking_meta_learner(
        stack_train_prob_matrix, y_train, stack_val_prob_matrix
    )
    thr_E, metrics_E = choose_threshold_f1(y_val, stack_meta_prob)
    elapsed = time.time() - t0
    print_metrics("方向E 4模型Stacking(LR)", metrics_E, elapsed)
    results.append({"method": "【方向E】4模型Stacking ET+XGB+ExtET+HistGB", **metrics_E, "seconds": round(elapsed, 1)})

    # ══════════════════════════════════════════════════════════════════════════
    # 【方向 F】F-beta 阈值优化（原有方案 + F0.5 阈值，提高精确率）
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[方向 F] F-beta(0.5) 阈值优化（基线概率，提高精确率）", flush=True)
    thr_F05, metrics_F05 = choose_threshold_fbeta(y_val, baseline_prob, beta=0.5)
    print_metrics("方向F F-beta(0.5) 基线概率", metrics_F05)
    results.append({"method": "【方向F】F-beta(0.5) 基线概率（高精确率）", **metrics_F05, "seconds": 0.0})

    print("\n[方向 F] F-beta(2.0) 阈值优化（基线概率，提高召回率/鲁棒性）", flush=True)
    thr_F2, metrics_F2 = choose_threshold_fbeta(y_val, baseline_prob, beta=2.0)
    print_metrics("方向F F-beta(2.0) 基线概率", metrics_F2)
    results.append({"method": "【方向F】F-beta(2.0) 基线概率（高召回率）", **metrics_F2, "seconds": 0.0})

    # ══════════════════════════════════════════════════════════════════════════
    # 【最优组合】选取验证集 F1 最高的方案生成最终预测
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("[结果汇总] 所有方向验证集性能对比")
    print("=" * 70)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(results_df[["method", "f1", "precision", "recall", "ap", "mcc", "threshold", "seconds"]].to_string(index=False))
    results_df.to_csv(EXPERIMENT_CSV, index=False)
    print(f"\n已写入 {EXPERIMENT_CSV}")

    # ── 生成最终预测（使用方向 D 或 E 中表现最佳的方案）──────────────────────
    # 根据验证集结果，选 F1 最高的方案用于测试集预测
    best_row = results_df.iloc[0]
    print(f"\n最佳方案: {best_row['method']}  F1={best_row['f1']:.4f}", flush=True)

    # 重训全量数据（示例：用扩展特征 ET + XGBoost，该组合最常见最优）
    print("\n[最终预测] 用扩展特征 ET 0.75 + XGBoost 0.25 重训全量数据...", flush=True)
    spw_full = float((len(y) - y.sum()) / max(1, y.sum()))
    final_et = make_et_model()
    final_xgb = make_xgb_model(spw_full)
    final_et.fit(x_ext, y)
    final_xgb.fit(x_ext, y)

    # Task1（test_simple）使用 F1 最大化阈值
    simple_prob = 0.75 * final_et.predict_proba(x_simple_ext)[:, 1] + \
                  0.25 * final_xgb.predict_proba(x_simple_ext)[:, 1]
    # Task2（test_complex）使用 F-beta(2.0) 阈值，更重视召回率（提升鲁棒性）
    complex_prob = 0.75 * final_et.predict_proba(x_complex_ext)[:, 1] + \
                   0.25 * final_xgb.predict_proba(x_complex_ext)[:, 1]

    # Task1 用验证集最优 F1 阈值
    thr_simple = thr_A
    # Task2 用更低阈值（F-beta=2.0 思路：宁可多报，不能漏报，提高鲁棒性）
    _, metrics_task2 = choose_threshold_fbeta(y_val, ext_blend_prob, beta=1.5)
    thr_complex = metrics_task2["threshold"]
    print(f"Task1 阈值（F1最优）: {thr_simple:.6f}", flush=True)
    print(f"Task2 阈值（F-beta=1.5）: {thr_complex:.6f}", flush=True)

    pred_simple = (simple_prob >= thr_simple).astype(np.int8)
    pred_complex = (complex_prob >= thr_complex).astype(np.int8)

    pd.DataFrame({"y_pred": pred_simple}).to_csv(PRED_SIMPLE_NEW, index=False)
    pd.DataFrame({"y_pred": pred_complex}).to_csv(PRED_COMPLEX_NEW, index=False)
    print(f"写出 {PRED_SIMPLE_NEW}: {len(pred_simple)} 行, 预测正例={int(pred_simple.sum())}", flush=True)
    print(f"写出 {PRED_COMPLEX_NEW}: {len(pred_complex)} 行, 预测正例={int(pred_complex.sum())}", flush=True)


if __name__ == "__main__":
    main()
