"""Walk-forward backtest: fixed ET+XGBoost temporal blend across time windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from compare_lgbm_xgboost import (
    build_temporal_features,
    choose_threshold,
    make_et_temporal_model,
    make_xgb_temporal_model,
    positive_proba,
)

# 与 train_predict / 主验证融合可对齐的 ET 权重（XGBoost 为 1 - 该值）
BLEND_ET_WEIGHT = 0.80
WALK_FORWARD_CSV = "walk_forward_comparison.csv"


def main() -> None:
    train = pd.read_csv("train.csv")
    base_cols = [col for col in train.columns if col != "y"]
    y = train["y"].astype(np.int8).to_numpy()

    print("Building temporal features...", flush=True)
    x_temporal = build_temporal_features(train, base_cols)

    # 固定验证跨度 4%，训练起点滑动
    train_starts = [0.91, 0.92, 0.93, 0.94, 0.95]
    windows = [(r, r + 0.04) for r in train_starts]
    wf_rows: list[dict[str, float | str | int]] = []
    print("\nRunning walk-forward backtest...", flush=True)

    for train_ratio, val_ratio in windows:
        wf_train_end = int(len(train) * train_ratio)
        wf_val_end = int(len(train) * val_ratio)
        wf_y_train = y[:wf_train_end]
        wf_y_val = y[wf_train_end:wf_val_end]
        wf_scale_pos_weight = float(
            (len(wf_y_train) - wf_y_train.sum()) / max(1, wf_y_train.sum())
        )
        window_name = f"0~{int(train_ratio * 100)} -> {int(train_ratio * 100)}~{int(val_ratio * 100)}"
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

        wf_et = make_et_temporal_model()
        wf_xgb = make_xgb_temporal_model(wf_scale_pos_weight)
        wf_et.fit(x_temporal.iloc[:wf_train_end], wf_y_train)
        wf_xgb.fit(x_temporal.iloc[:wf_train_end], wf_y_train)

        wf_et_prob = positive_proba(wf_et, x_temporal.iloc[wf_train_end:wf_val_end])
        wf_xgb_prob = positive_proba(wf_xgb, x_temporal.iloc[wf_train_end:wf_val_end])
        wf_blend_prob = BLEND_ET_WEIGHT * wf_et_prob + (1.0 - BLEND_ET_WEIGHT) * wf_xgb_prob
        wf_metrics = choose_threshold(wf_y_val, wf_blend_prob)
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
