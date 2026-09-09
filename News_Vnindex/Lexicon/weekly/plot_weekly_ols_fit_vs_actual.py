"""Biểu đồ đường: giá trị THỰC vs OLS DỰ BÁO (fitted) theo thời gian, cho
đúng 8 tổ hợp có adjusted R² ÂM đã phát hiện (horizon 1 tuần, cả 4 method) -
xem trực quan mức độ "không học được gì" tương ứng với R² âm.

Mirror plot_ols_fit_vs_actual.py (bản daily) nhưng cho weekly - vì weekly
chỉ ~710-722 tuần (khác daily ~3.350 ngày) nên vẽ được TOÀN BỘ giai đoạn,
không cần zoom vào 1 khung thời gian ngắn.

Import lại PREDICTOR_COLUMNS/ABNORMAL_INPUT_PATHS_BY_METHOD từ
vnindex_weekly_predictive_regression.py (bản đơn giản, CÙNG thư mục weekly/)
- không viết lại danh sách predictor.

Output: data_News/weekly_ols_fit_vs_actual.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnindex_weekly_predictive_regression import (  # noqa: E402
    ABNORMAL_INPUT_PATHS_BY_METHOD,
    PREDICTOR_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "weekly_ols_fit_vs_actual.png"

# Đúng 8 tổ hợp có adjusted R² âm đã phát hiện (horizon 1 tuần)
TARGET_COLUMNS = ["future_ret_1w", "future_abnormal_ar1_ret_1w"]

COLOR_ACTUAL = "#1d4ed8"
COLOR_FITTED = "#ea580c"


def fit_ols(model_df: pd.DataFrame, target_column: str, predictor_columns: list[str]):
    y = model_df[target_column].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(model_df)), model_df[predictor_columns].to_numpy(dtype="float64")])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficients
    residuals = y - fitted
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss
    n, k = x.shape
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k)
    return fitted, r_squared, adj_r_squared


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    method_names = list(ABNORMAL_INPUT_PATHS_BY_METHOD.keys())
    fig, axes = plt.subplots(len(method_names), len(TARGET_COLUMNS), figsize=(14, 12), sharex=True)

    for row_index, method_name in enumerate(method_names):
        merged_df = pd.read_parquet(ABNORMAL_INPUT_PATHS_BY_METHOD[method_name])
        merged_df = merged_df.sort_values("week_end").reset_index(drop=True)

        for col_index, target_column in enumerate(TARGET_COLUMNS):
            existing_predictors = [column for column in PREDICTOR_COLUMNS if column in merged_df.columns]
            model_df = merged_df[["week_end", target_column] + existing_predictors].dropna()

            fitted, r_squared, adj_r_squared = fit_ols(model_df, target_column, existing_predictors)

            ax = axes[row_index, col_index]
            ax.plot(model_df["week_end"], model_df[target_column] * 10000, color=COLOR_ACTUAL, linewidth=1.0, label="Thực tế")
            ax.plot(model_df["week_end"], fitted * 10000, color=COLOR_FITTED, linewidth=1.4, label="OLS dự báo")
            ax.axhline(0, color="#334155", linewidth=0.7, linestyle="--")
            ax.set_title(
                f"{method_name} / {target_column}\nR²={r_squared:.4f}  adj R²={adj_r_squared:.4f}",
                fontsize=9,
            )
            ax.tick_params(labelsize=7)
            if row_index == 0 and col_index == 0:
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            if col_index == 0:
                ax.set_ylabel("Return (bp)", fontsize=8)

    fig.suptitle("Thực tế vs OLS dự báo - 8 tổ hợp có adjusted R² âm (horizon 1 tuần)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print("Đã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
