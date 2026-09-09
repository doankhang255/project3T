"""Vẽ biểu đồ hệ số sentiment theo từng lag (1-5) kèm khoảng tin cậy 95%
(Newey-West) - kiểu "event study"/coefficient plot chuẩn trong tài chính
(giống cách Tetlock trình bày Table II trong REF) - để nhìn trực quan tác
động của sentiment lên lợi suất VN-Index, thay vì chỉ đọc bảng số.

Small-multiples 2x2 (hàng = method Cách 1/Cách 2, cột = target rolling/AR1)
- mỗi ô: trục x = lag (1-5 ngày), trục y = hệ số (đổi sang basis point,
x10000, cùng đơn vị Tetlock dùng để báo cáo), thanh sai số = 1,96 x SE
Newey-West (khoảng tin cậy 95%). Điểm có ý nghĩa (p<0,05, khoảng tin cậy
không chứa 0) tô đậm/đổi màu khác điểm không có ý nghĩa.

Import lại add_regression_features/fit_with_lag_sum_test từ file gốc
(không viết lại logic hồi quy).

Output: data_News/sentiment_lag_coefficients.png
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
from vnindex_daily_predictive_regression import (  # noqa: E402
    MERGED_DATA_PATHS_BY_METHOD,
    N_LAGS,
    SENTIMENT_COLUMN,
    TARGET_COLUMNS,
    add_regression_features,
    fit_with_lag_sum_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "sentiment_lag_coefficients.png"

# Màu trung tính (không ý nghĩa) vs màu nhấn (có ý nghĩa, p<0.05) - 1 cặp
# diverging đơn giản, không dùng màu tùy tiện.
COLOR_NOT_SIGNIFICANT = "#94a3b8"  # xám ánh xanh
COLOR_SIGNIFICANT = "#dc2626"  # đỏ
COLOR_ZERO_LINE = "#334155"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    method_names = list(MERGED_DATA_PATHS_BY_METHOD.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

    for row_index, method_name in enumerate(method_names):
        merged_df = pd.read_parquet(MERGED_DATA_PATHS_BY_METHOD[method_name])
        for col_index, target_column in enumerate(TARGET_COLUMNS):
            featured_df, sentiment_lag_columns, predictor_columns = add_regression_features(
                merged_df, target_column
            )
            coef_df, lag_sum_result = fit_with_lag_sum_test(
                featured_df, target_column, predictor_columns, sentiment_lag_columns
            )

            sentiment_rows = coef_df.loc[
                coef_df["predictor_variable"].str.startswith(f"{SENTIMENT_COLUMN}_lag")
            ].copy()
            sentiment_rows["lag"] = range(1, N_LAGS + 1)
            sentiment_rows["coef_bp"] = sentiment_rows["coefficient"] * 10000
            sentiment_rows["ci_bp"] = 1.96 * sentiment_rows["std_error_newey_west"] * 10000
            sentiment_rows["significant"] = sentiment_rows["p_value"] < 0.05

            ax = axes[row_index, col_index]
            colors = [
                COLOR_SIGNIFICANT if sig else COLOR_NOT_SIGNIFICANT
                for sig in sentiment_rows["significant"]
            ]
            ax.axhline(0, color=COLOR_ZERO_LINE, linewidth=1, linestyle="--", zorder=1)
            for lag_value, coef_value, ci_value, color in zip(
                sentiment_rows["lag"], sentiment_rows["coef_bp"], sentiment_rows["ci_bp"], colors
            ):
                ax.errorbar(
                    [lag_value],
                    [coef_value],
                    yerr=[[ci_value], [ci_value]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=4,
                    zorder=2,
                )
            ax.scatter(sentiment_rows["lag"], sentiment_rows["coef_bp"], c=colors, s=55, zorder=3)

            reversal_p = lag_sum_result["reversal_lag2_5_sum_p"]
            ax.set_title(
                f"{method_name} / {target_column}\n"
                f"lag1 p={coef_df.loc[coef_df['predictor_variable']==sentiment_lag_columns[0],'p_value'].iloc[0]:.2f}"
                f"  |  tổng lag2-5 p={reversal_p:.2f}",
                fontsize=9,
            )
            ax.set_xticks(range(1, N_LAGS + 1))
            ax.set_xlabel("Lag (ngày)", fontsize=9)
            if col_index == 0:
                ax.set_ylabel("Hệ số sentiment (basis point)", fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_SIGNIFICANT, linestyle="", markersize=7, label="p < 0,05"),
        plt.Line2D(
            [0], [0], marker="o", color=COLOR_NOT_SIGNIFICANT, linestyle="", markersize=7, label="p ≥ 0,05"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Tác động của sentiment (5 lag) lên abnormal return VN-Index - khoảng tin cậy 95% (Newey-West)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print("Đã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
