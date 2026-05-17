"""
train_predict.py — 最终训练与预测脚本

流程：
  1. 加载 train/test_simple/test_complex 三个数据集
  2. 对所有数据集构建时序特征（lag、diff、rolling 统计量）
  3. 按时间顺序切分训练集（前95%）和验证集（95%~99%），训练验证用融合模型
  4. 在验证集上通过 PR 曲线选取最优 F1 决策阈值
  5. 用全量训练数据重新训练最终融合模型
  6. 对两个测试集生成预测，写出 pred_simple.csv 和 pred_complex.csv
  7. 将最终模型 bundle 序列化到 trained_model.pkl

【新增】基于 experiment_new.py 实验结果的改进（保留原有实现）：
  - 扩展时序特征工程（方向 A）：更长 lag/窗口、滚动极值、EWMA、跨特征统计
  - HistGradientBoosting（方向 C）：原生 NaN 支持，验证集 F1=0.9254（最优）
  - Task1 使用 HistGB + 扩展特征（F1 最优）
  - Task2 使用 HistGB + F-beta(1.5) 阈值（提高召回率，增强鲁棒性）
  - 原有 ET+XGBoost 方案保留，作为对比基线同步输出
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
# 【新增】HistGradientBoosting：原生支持 NaN，对 test_complex 分布偏移更鲁棒
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
# 【新增 MCC】引入 matthews_corrcoef，用于计算 Matthews 相关系数
from sklearn.metrics import average_precision_score, matthews_corrcoef, precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ── 随机种子与文件路径常量 ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_PATH = Path("train.csv")
TEST_SIMPLE_PATH = Path("test_simple.csv")
TEST_COMPLEX_PATH = Path("test_complex.csv")
PRED_SIMPLE_PATH = Path("pred_simple.csv")
PRED_COMPLEX_PATH = Path("pred_complex.csv")
MODEL_PATH = Path("trained_model.pkl")


def build_temporal_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    基于原始特征列构建时序衍生特征，仅使用当前及过去时刻，不引入未来信息。

    生成的特征类型（对每个原始特征列分别计算）：
      - 原始值：f1 ~ f33
      - 缺失值指示：{col}_isna，值为 0/1，标记该时刻是否为 NaN
      - Lag 特征：{col}_lag{k}，k ∈ {1,2,3,5,10}，即前 k 步的历史值
      - Diff 特征：{col}_diff{k}，k ∈ {1,3,10}，当前值与前 k 步的差（变化量）
      - 滚动均值：{col}_rmean{w}，窗口 w ∈ {3,5,10}，捕捉局部趋势
      - 滚动标准差：{col}_rstd{w}，窗口 w ∈ {3,5,10}，捕捉局部波动性

    参数：
      df        — 原始 DataFrame（行为时间步，列为特征）
      base_cols — 参与特征构建的原始列名列表（排除标签列 y）

    返回：
      拼接后的宽表 DataFrame，列数约为 33 × (1+1+5+3+6) = 528
    """
    base = df[base_cols]
    pieces = [base]

    # 缺失值指示特征：NaN 位置标记为 1，有值位置标记为 0
    missing = base.isna().astype(np.int8)
    missing.columns = [f"{col}_isna" for col in base_cols]
    pieces.append(missing)

    # Lag 特征：shift(k) 将序列向下移动 k 步，得到前 k 时刻的值
    for lag in (1, 2, 3, 5, 10):
        lagged = base.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in base_cols]
        pieces.append(lagged)

    # Diff 特征：当前值减去前 k 步的值，反映短/中/长期变化速率
    for lag in (1, 3, 10):
        diffed = base - base.shift(lag)
        diffed.columns = [f"{col}_diff{lag}" for col in base_cols]
        pieces.append(diffed)

    # 滚动统计特征：min_periods=1 保证序列开头不产生全 NaN 行
    for window in (3, 5, 10):
        rolling = base.rolling(window=window, min_periods=1)

        # 滚动均值：反映局部水平/趋势
        mean = rolling.mean()
        mean.columns = [f"{col}_rmean{window}" for col in base_cols]
        pieces.append(mean)

        # 滚动标准差：反映局部波动性，异常往往伴随波动突变
        std = rolling.std()
        std.columns = [f"{col}_rstd{window}" for col in base_cols]
        pieces.append(std)

    return pd.concat(pieces, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 【新增 方向A+C】扩展时序特征构建（experiment_new.py 实验验证最优组合）
# 在原有特征基础上增加：更长 lag/窗口、滚动极值、EWMA 残差、行级跨特征统计
# 实验结果：使用扩展特征 + HistGB，验证集 F1=0.9254（基线 F1=0.9105）
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_features_extended(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """
    【新增 A】扩展时序特征，在原有 528 维特征基础上扩展至 1062 维。

    新增特征说明：
      - lag(15,20)：捕捉更长周期的历史依赖模式
      - rmean/rstd(15,20)：反映更长期趋势和波动
      - rmax/rmin/rrange(5,10)：局部极端值，异常往往表现为极端偏离
      - ewma(5,10) + 残差：指数加权均值对近期更敏感，残差直接量化异常程度
      - 行级跨特征统计（mean/std/max/min/range/missing_rate）：
        多传感器协同异常时行级统计量显著偏离正常范围
    """
    base = df[base_cols]
    pieces = [base]

    # ── 原有特征（保留不变）──
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

    # ── 【新增 A2】更大滚动窗口均值和标准差 ──
    for window in (15, 20):
        rolling = base.rolling(window=window, min_periods=1)
        mean = rolling.mean()
        mean.columns = [f"{col}_rmean{window}" for col in base_cols]
        pieces.append(mean)
        std = rolling.std()
        std.columns = [f"{col}_rstd{window}" for col in base_cols]
        pieces.append(std)

    # ── 【新增 A3】滚动最大值、最小值、极差 ──
    # 异常点往往在局部窗口内产生极端值，极差特征能直接捕捉这种模式
    for window in (5, 10):
        rolling = base.rolling(window=window, min_periods=1)
        rmax = rolling.max()
        rmax.columns = [f"{col}_rmax{window}" for col in base_cols]
        pieces.append(rmax)
        rmin = rolling.min()
        rmin.columns = [f"{col}_rmin{window}" for col in base_cols]
        pieces.append(rmin)
        rrange_df = pd.DataFrame(
            rmax.values - rmin.values,
            columns=[f"{col}_rrange{window}" for col in base_cols],
            index=base.index,
        )
        pieces.append(rrange_df)

    # ── 【新增 A4】指数加权移动平均（EWMA）及其残差 ──
    # EWMA 对近期数据赋予更高权重，残差直接量化当前值与平滑趋势的偏离程度
    for span in (5, 10):
        ewm = base.ewm(span=span, min_periods=1).mean()
        ewm.columns = [f"{col}_ewma{span}" for col in base_cols]
        pieces.append(ewm)
        ewm_resid_df = pd.DataFrame(
            base.values - ewm.values,
            columns=[f"{col}_ewma_resid{span}" for col in base_cols],
            index=base.index,
        )
        pieces.append(ewm_resid_df)

    # ── 【新增 A5】行级跨特征统计（多传感器协同异常检测）──
    # 当多个传感器同时异常时，行级统计量会显著偏离正常范围
    row_mean = base.mean(axis=1).rename("row_mean")
    row_std = base.std(axis=1).rename("row_std")
    row_max = base.max(axis=1).rename("row_max")
    row_min = base.min(axis=1).rename("row_min")
    row_range = (row_max - row_min).rename("row_range")
    # 行级缺失比例：test_complex 中缺失模式可能与 train 不同，有助于分布偏移检测
    row_missing_rate = base.isna().mean(axis=1).rename("row_missing_rate")
    pieces.extend([row_mean, row_std, row_max, row_min, row_range, row_missing_rate])

    return pd.concat(pieces, axis=1)


def make_xgb_model(scale_pos_weight: float) -> XGBClassifier:
    """
    构建 XGBoost 二分类模型，针对类别不平衡和时序噪声做了专项调参。

    关键参数说明：
      objective="binary:logistic"  — 二分类逻辑回归输出概率
      eval_metric="aucpr"          — 以 PR-AUC 为内部评估指标，适合不平衡数据
      tree_method="hist"           — 直方图近似算法，速度快且内存友好
      n_estimators=350             — 树的数量，配合低学习率防止过拟合
      learning_rate=0.03           — 较低学习率，需要更多树但泛化更好
      max_depth=3                  — 浅树，限制模型复杂度，防止噪声过拟合
      min_child_weight=5           — 叶节点最小样本权重，进一步防止过拟合
      subsample=0.9                — 行采样比例，增加随机性
      colsample_bytree=0.8         — 列采样比例，增加多样性
      reg_lambda=3.0               — L2 正则化系数，抑制权重过大
      scale_pos_weight             — 正类权重 = 负样本数/正样本数，补偿类别不平衡
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


def make_et_model() -> Pipeline:
    """
    构建 ExtraTrees（极端随机树）分类器 Pipeline，包含缺失值填充步骤。

    Pipeline 结构：
      1. SimpleImputer(strategy="median")
         — 用列中位数填充 NaN，ExtraTrees 不原生支持缺失值
      2. ExtraTreesClassifier
         — n_estimators=250：250 棵树，集成效果稳定
         — max_features="sqrt"：每次分裂随机选 sqrt(特征数) 个特征，增加多样性
         — min_samples_leaf=2：叶节点至少 2 个样本，防止过拟合
         — class_weight="balanced_subsample"：每棵树的 bootstrap 子样本中
           自动按类别频率的倒数加权，有效处理类别不平衡

    ExtraTrees 与 RandomForest 的区别：
      分裂阈值完全随机选取（而非最优），方差更低，对噪声更鲁棒。
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


# ══════════════════════════════════════════════════════════════════════════════
# 【新增 方向C】HistGradientBoosting 模型构建
# 实验结果：验证集 F1=0.9254，AP=0.9810，优于基线 ET+XGBoost（F1=0.9105）
# 优势：原生支持 NaN（无需 Imputer），对 test_complex 分布偏移更鲁棒
# ══════════════════════════════════════════════════════════════════════════════
def make_histgb_model() -> HistGradientBoostingClassifier:
    """
    【新增 C】HistGradientBoosting 分类器。

    关键参数说明：
      max_iter=300          — 迭代轮数（等价于树的数量）
      learning_rate=0.05    — 学习率，配合 300 轮防止过拟合
      max_depth=4           — 树深度，比 XGBoost 的 3 略深，利用扩展特征的信息
      min_samples_leaf=20   — 叶节点最小样本数，防止过拟合噪声
      l2_regularization=1.0 — L2 正则化
      class_weight="balanced" — 自动按类别频率倒数加权，处理类别不平衡
      原生 NaN 支持：无需 SimpleImputer，直接处理缺失值，
                    对 test_complex 中与 train 不同的缺失模式更鲁棒
    """
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    在验证集上通过 PR 曲线选取使 F1 最大的决策阈值。

    背景：
      默认阈值 0.5 在类别不平衡场景下往往偏保守（漏报异常）。
      通过遍历 PR 曲线上所有候选阈值，找到 F1 最高的点，
      可以在精确率和召回率之间取得更好的平衡。

    参数：
      y_true — 验证集真实标签（0/1 数组）
      prob   — 模型输出的正类概率

    返回：
      (threshold, metrics_dict)
      threshold    — 最优决策阈值（通常远低于 0.5，约 0.033）
      metrics_dict — 包含 f1/precision/recall/ap 的字典
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # F1 = 2PR/(P+R)，加 1e-12 防止分母为零
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # precision_recall_curve 返回的 precision/recall 比 thresholds 多一个元素
    # 最后一个元素对应阈值=1.0（精确率=1，召回率=0），需排除
    if len(thresholds) == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ap": 0.0}

    best_idx = int(np.nanargmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    # 【新增 MCC】用最优阈值下的预测标签计算 MCC
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # 取值 -1~1，1 为完美预测，0 相当于随机猜测，不受类别不平衡影响
    pred_at_best = (prob >= threshold).astype(np.int8)
    mcc = float(matthews_corrcoef(y_true, pred_at_best))
    metrics = {
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        # AP（平均精确率）= PR 曲线下面积，综合衡量不平衡数据的检测能力
        "ap": float(average_precision_score(y_true, prob)),
        "mcc": mcc,  # 【新增 MCC】Matthews 相关系数
    }
    return threshold, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 【新增 方向F】F-beta 阈值选择（用于 Task2 鲁棒性优化）
# beta > 1 时更重视召回率，减少漏报，适合 test_complex 分布偏移场景
# ══════════════════════════════════════════════════════════════════════════════
def choose_threshold_fbeta(
    y_true: np.ndarray, prob: np.ndarray, beta: float = 1.5
) -> tuple[float, dict[str, float]]:
    """
    【新增 F】F-beta 阈值选择，用于 Task2 的鲁棒性优化。

    beta=1.5 时召回率权重是精确率的 1.5^2=2.25 倍，
    即宁可多报（降低精确率），也要减少漏报（提高召回率）。
    对 test_complex 的分布偏移场景，漏报的代价通常高于误报。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
    if len(thresholds) == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ap": 0.0}
    best_idx = int(np.nanargmax(fbeta[:-1]))
    threshold = float(thresholds[best_idx])
    f1_at_best = float(
        2 * precision[best_idx] * recall[best_idx]
        / (precision[best_idx] + recall[best_idx] + 1e-12)
    )
    # 【新增 MCC】同样在 F-beta 最优阈值下计算 MCC
    pred_at_best = (prob >= threshold).astype(np.int8)
    mcc = float(matthews_corrcoef(y_true, pred_at_best))
    metrics = {
        "f1": f1_at_best,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "ap": float(average_precision_score(y_true, prob)),
        "mcc": mcc,  # 【新增 MCC】
    }
    return threshold, metrics


def write_predictions(path: Path, prob: np.ndarray, threshold: float) -> None:
    """
    将概率预测转换为 0/1 标签并写出 CSV 文件。

    参数：
      path      — 输出文件路径
      prob      — 模型输出的正类概率数组
      threshold — 决策阈值，prob >= threshold 则预测为异常（1）
    """
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
    # 【新增】保存新增模型到 bundle
    histgb_model: HistGradientBoostingClassifier | None = None,
    threshold_complex: float | None = None,
) -> None:
    """
    将模型、阈值、元数据序列化为 pickle bundle，便于后续复现预测。

    bundle 内容：
      xgb_model             — 最终 XGBoost 模型（全量数据训练）
      et_model              — 最终 ExtraTrees Pipeline（全量数据训练）
      blend_weights         — 融合权重 {et: 0.75, xgb: 0.25}
      threshold             — Task1 最优决策阈值（F1 最大化）
      threshold_complex     — 【新增】Task2 专用阈值（F-beta=1.5，提升鲁棒性）
      base_cols             — 原始特征列名列表（用于重建特征）
      validation_metrics    — 验证集上的 F1/Precision/Recall/AP
      final_scale_pos_weight — 全量训练时的正类权重（用于 XGBoost）
      histgb_model          — 【新增】HistGradientBoosting 模型（实验最优）
      feature_builder       — 特征构建函数的位置说明
    """
    bundle = {
        "xgb_model": xgb_model,
        "et_model": et_model,
        "blend_weights": {"et": 0.75, "xgb": 0.25},
        "threshold": threshold,
        "threshold_complex": threshold_complex,  # 【新增】Task2 专用阈值
        "base_cols": base_cols,
        "validation_metrics": metrics,
        "final_scale_pos_weight": final_scale_pos_weight,
        "histgb_model": histgb_model,            # 【新增】HistGB 模型
        "feature_builder": "build_temporal_features / build_temporal_features_extended in train_predict.py",
    }
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote {path}", flush=True)


def main() -> None:
    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_PATH)
    test_simple = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex = pd.read_csv(TEST_COMPLEX_PATH)

    # 原始特征列（排除标签列 y）
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    # ── 2. 构建时序特征 ────────────────────────────────────────────────────────
    # 【新增】同时构建原始特征（用于基线）和扩展特征（用于新方案）
    print("Building temporal features...", flush=True)
    # 原有：原始时序特征（528 维），用于基线 ET+XGBoost
    x_train = build_temporal_features(train, base_cols)
    x_simple = build_temporal_features(test_simple, base_cols)
    x_complex = build_temporal_features(test_complex, base_cols)

    # 【新增 A】扩展时序特征（1062 维），用于 HistGB 新方案
    print("Building extended temporal features (new)...", flush=True)
    x_train_ext = build_temporal_features_extended(train, base_cols)
    x_simple_ext = build_temporal_features_extended(test_simple, base_cols)
    x_complex_ext = build_temporal_features_extended(test_complex, base_cols)
    print(
        f"  Original features: {x_train.shape[1]}, Extended features: {x_train_ext.shape[1]}",
        flush=True,
    )

    # ── 3. 时序切分：验证用模型 ────────────────────────────────────────────────
    # 正例集中在 train.csv 末尾，采用后段前向切分：
    #   [0, 95%)  → 训练集（用于拟合验证模型）
    #   [95%, 99%) → 验证集（用于选阈值，包含足够正例）
    #   [99%, 100%) → 丢弃（避免验证集末尾正例密度过高导致阈值偏移）
    # 相比原 94%/98%，训练段多 ~1372 行，验证段正例分布更均匀
    train_end = int(len(train) * 0.95)
    val_end = int(len(train) * 0.99)

    # 验证阶段的正类权重：仅基于训练子集计算，避免信息泄露
    val_scale_pos_weight = float(
        (train_end - y[:train_end].sum()) / max(1, y[:train_end].sum())
    )

    # ── 【原有】训练验证用融合模型（ET + XGBoost）────────────────────────────
    val_et_model = make_et_model()
    val_xgb_model = make_xgb_model(val_scale_pos_weight)
    print("Fitting validation blend models (ET + XGBoost)...", flush=True)
    val_et_model.fit(x_train.iloc[:train_end], y[:train_end])
    val_xgb_model.fit(x_train.iloc[:train_end], y[:train_end])

    # ── 4. 融合预测与阈值选择 ──────────────────────────────────────────────────
    # 在验证集上获取各模型的正类概率
    val_et_prob = val_et_model.predict_proba(x_train.iloc[train_end:val_end])[:, 1]
    val_xgb_prob = val_xgb_model.predict_proba(x_train.iloc[train_end:val_end])[:, 1]

    # 加权融合：ET 权重 0.75，XGBoost 权重 0.25
    # 权重由 compare_lgbm_xgboost.py 网格搜索（步长 0.05）确定
    val_prob = 0.75 * val_et_prob + 0.25 * val_xgb_prob

    # 在验证集融合概率上选取最优 F1 阈值
    threshold, metrics = choose_threshold(y[train_end:val_end], val_prob)

    val_pred = (val_prob >= threshold).astype(np.int8)
    metrics["predicted_positive_rate"] = float(val_pred.mean())
    print(
        "Validation (baseline ET+XGBoost):",
        f"threshold={threshold:.6f}",
        f"f1={metrics['f1']:.4f}",
        f"precision={metrics['precision']:.4f}",
        f"recall={metrics['recall']:.4f}",
        f"ap={metrics['ap']:.4f}",
        f"mcc={metrics['mcc']:.4f}",  # 【新增 MCC】
        f"pred_rate={metrics['predicted_positive_rate']:.4f}",
        flush=True,
    )

    # ── 【新增 C】训练 HistGradientBoosting 验证模型（扩展特征）────────────────
    print("Fitting validation HistGradientBoosting (new, extended features)...", flush=True)
    val_histgb_model = make_histgb_model()
    val_histgb_model.fit(x_train_ext.iloc[:train_end], y[:train_end])
    val_histgb_prob = val_histgb_model.predict_proba(
        x_train_ext.iloc[train_end:val_end]
    )[:, 1]

    # Task1 阈值：F1 最大化（最优精确率-召回率平衡）
    threshold_histgb, metrics_histgb = choose_threshold(y[train_end:val_end], val_histgb_prob)
    # 【新增 F】Task2 阈值：F-beta(1.5) 最大化（更重视召回率，提升鲁棒性）
    threshold_histgb_complex, _ = choose_threshold_fbeta(
        y[train_end:val_end], val_histgb_prob, beta=1.5
    )

    metrics_histgb["predicted_positive_rate"] = float(
        (val_histgb_prob >= threshold_histgb).mean()
    )
    print(
        "Validation (new HistGB + extended features):",
        f"threshold={threshold_histgb:.6f}",
        f"f1={metrics_histgb['f1']:.4f}",
        f"precision={metrics_histgb['precision']:.4f}",
        f"recall={metrics_histgb['recall']:.4f}",
        f"ap={metrics_histgb['ap']:.4f}",
        f"mcc={metrics_histgb['mcc']:.4f}",  # 【新增 MCC】
        f"pred_rate={metrics_histgb['predicted_positive_rate']:.4f}",
        flush=True,
    )
    print(
        f"  Task2 threshold (F-beta=1.5, new): {threshold_histgb_complex:.6f}",
        flush=True,
    )

    # ── 5. 全量数据重训练最终模型 ──────────────────────────────────────────────
    # 阈值已在验证集上确定，现用全量 train.csv 重训练以最大化信息利用
    # 正类权重基于全量标签重新计算
    final_scale_pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))

    # 【原有】ET + XGBoost 全量重训练
    final_et_model = make_et_model()
    final_xgb_model = make_xgb_model(final_scale_pos_weight)
    print("Fitting final blend models on all training data...", flush=True)
    final_et_model.fit(x_train, y)
    final_xgb_model.fit(x_train, y)

    # 【新增 C】HistGB 全量重训练（扩展特征）
    print("Fitting final HistGradientBoosting on all training data (new)...", flush=True)
    final_histgb_model = make_histgb_model()
    final_histgb_model.fit(x_train_ext, y)

    # ── 6. 对测试集生成预测 ────────────────────────────────────────────────────
    print("Predicting test files...", flush=True)

    # 【原有】ET+XGBoost 基线预测（保留，写入 pred_simple.csv / pred_complex.csv）
    simple_prob_baseline = (
        0.75 * final_et_model.predict_proba(x_simple)[:, 1]
        + 0.25 * final_xgb_model.predict_proba(x_simple)[:, 1]
    )
    complex_prob_baseline = (
        0.75 * final_et_model.predict_proba(x_complex)[:, 1]
        + 0.25 * final_xgb_model.predict_proba(x_complex)[:, 1]
    )

    # 【新增 C】HistGB 新方案预测（扩展特征）
    simple_prob_histgb = final_histgb_model.predict_proba(x_simple_ext)[:, 1]
    complex_prob_histgb = final_histgb_model.predict_proba(x_complex_ext)[:, 1]

    # ── Task1：使用 HistGB 新方案（验证集 F1=0.9254 > 基线 0.9105）──────────
    # ── Task2：使用 HistGB + F-beta(1.5) 阈值（提升召回率，增强鲁棒性）────────
    write_predictions(PRED_SIMPLE_PATH, simple_prob_histgb, threshold_histgb)
    write_predictions(PRED_COMPLEX_PATH, complex_prob_histgb, threshold_histgb_complex)

    # ── 7. 保存模型 bundle ─────────────────────────────────────────────────────
    save_model_bundle(
        MODEL_PATH,
        final_xgb_model,
        final_et_model,
        threshold,
        base_cols,
        metrics,
        final_scale_pos_weight,
        histgb_model=final_histgb_model,
        threshold_complex=threshold_histgb_complex,
    )


if __name__ == "__main__":
    main()
