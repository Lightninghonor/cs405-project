"""
walk_forward_eval.py — 滚动前向验证（Walk-Forward Backtest）脚本

目的：
  验证 ET+XGBoost 融合模型在不同训练窗口下的稳定性。
  通过滑动训练起点（91%~95%），固定验证跨度（4%），
  观察模型性能是否随时间窗口变化而大幅波动。

验证窗口设计：
  训练起点  验证区间
  0~91%  → 91%~95%
  0~92%  → 92%~96%
  0~93%  → 93%~97%
  0~94%  → 94%~98%  ← 与 train_predict.py 主验证一致
  0~95%  → 95%~99%

输出：
  walk_forward_comparison.csv — 5 个窗口的 F1/Precision/Recall/AP/阈值等指标
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 复用 compare_lgbm_xgboost.py 中的工具函数，保持特征构建和模型配置一致
from compare_lgbm_xgboost import (
    build_temporal_features,
    choose_threshold,
    make_et_temporal_model,
    make_xgb_temporal_model,
    positive_proba,
)

# ET 权重（XGBoost 权重 = 1 - BLEND_ET_WEIGHT = 0.20）
# 注：此处使用 0.80 而非最终方案的 0.75，用于独立验证融合鲁棒性
BLEND_ET_WEIGHT = 0.80
WALK_FORWARD_CSV = "walk_forward_comparison.csv"


def main() -> None:
    # ── 1. 加载数据并构建时序特征 ──────────────────────────────────────────────
    train = pd.read_csv("train.csv")
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    print("Building temporal features...", flush=True)
    x_temporal = build_temporal_features(train, base_cols)

    # ── 2. 定义滚动窗口 ────────────────────────────────────────────────────────
    # 训练起点从 91% 滑动到 95%，验证跨度固定为 4%
    # 这样可以测试模型在不同数量正例下的稳定性
    train_starts = [0.91, 0.92, 0.93, 0.94, 0.95]
    windows = [(r, r + 0.04) for r in train_starts]
    wf_rows: list[dict[str, float | str | int]] = []
    print("\nRunning walk-forward backtest...", flush=True)

    for train_ratio, val_ratio in windows:
        wf_train_end = int(len(train) * train_ratio)
        wf_val_end = int(len(train) * val_ratio)
        wf_y_train = y[:wf_train_end]
        wf_y_val = y[wf_train_end:wf_val_end]

        # 每个窗口独立计算正类权重，基于该窗口的训练子集
        wf_scale_pos_weight = float(
            (len(wf_y_train) - wf_y_train.sum()) / max(1, wf_y_train.sum())
        )
        window_name = f"0~{int(train_ratio * 100)} -> {int(train_ratio * 100)}~{int(val_ratio * 100)}"

        # ── 边界检查：跳过训练集或验证集中只有单一类别的窗口 ──────────────────
        # 若训练集无正例，模型无法学习异常模式；
        # 若验证集无正例，无法计算 PR 曲线和 F1
        if np.unique(wf_y_train).size < 2 or np.unique(wf_y_val).size < 2:
            wf_rows.append(
                {
                    "window": window_name,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "ap": np.nan,
                    "threshold": np.nan,
                    "pred_rate": np.nan,
                    "train_size": int(wf_train_end),
                    "val_size": int(wf_val_end - wf_train_end),
                    "train_pos": int(wf_y_train.sum()),
                    "val_pos": int(wf_y_val.sum()),
                    "blend_method": f"{BLEND_ET_WEIGHT:.2f}*ET + {1.0 - BLEND_ET_WEIGHT:.2f}*XGBoost temporal",
                    "status": "skipped_single_class_window",
                }
            )
            continue

        # ── 3. 训练当前窗口的融合模型 ──────────────────────────────────────────
        wf_et = make_et_temporal_model()
        wf_xgb = make_xgb_temporal_model(wf_scale_pos_weight)
        wf_et.fit(x_temporal.iloc[:wf_train_end], wf_y_train)
        wf_xgb.fit(x_temporal.iloc[:wf_train_end], wf_y_train)

        # ── 4. 在验证集上融合预测并选阈值 ─────────────────────────────────────
        wf_et_prob = positive_proba(wf_et, x_temporal.iloc[wf_train_end:wf_val_end])
        wf_xgb_prob = positive_proba(wf_xgb, x_temporal.iloc[wf_train_end:wf_val_end])
        # 加权融合：ET 权重 0.80，XGBoost 权重 0.20
        wf_blend_prob = BLEND_ET_WEIGHT * wf_et_prob + (1.0 - BLEND_ET_WEIGHT) * wf_xgb_prob
        wf_metrics = choose_threshold(wf_y_val, wf_blend_prob)

        # 附加窗口元信息
        wf_metrics.update(
            {
                "window": window_name,
                "train_size": int(wf_train_end),
                "val_size": int(wf_val_end - wf_train_end),
                "train_pos": int(wf_y_train.sum()),
                "val_pos": int(wf_y_val.sum()),
                "blend_method": f"{BLEND_ET_WEIGHT:.2f}*ET + {1.0 - BLEND_ET_WEIGHT:.2f}*XGBoost temporal",
                "status": "ok",
            }
        )
        wf_rows.append(wf_metrics)

    # ── 5. 汇总输出 ────────────────────────────────────────────────────────────
    wf_out = pd.DataFrame(wf_rows).sort_values("train_size")
    wf_out.to_csv(WALK_FORWARD_CSV, index=False)
    print("\nWALK-FORWARD SUMMARY")
    print(
        wf_out[
            [
                "window",
                "f1",
                "precision",
                "recall",
                "ap",
                "threshold",
                "pred_rate",
                "train_pos",
                "val_pos",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {WALK_FORWARD_CSV}")


if __name__ == "__main__":
    main()
