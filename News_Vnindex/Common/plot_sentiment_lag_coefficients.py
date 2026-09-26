"""Vẽ biểu đồ hệ số sentiment theo từng lag (1-5) kèm khoảng tin cậy 95%
(Newey-West) - kiểu "event study"/coefficient plot chuẩn trong tài chính
(giống cách Tetlock trình bày Table II trong REF) - để nhìn trực quan tác
động của sentiment lên lợi suất VN-Index, thay vì chỉ đọc bảng số.

Small-multiples (hàng = từng method của nhánh, cột = target rolling/AR1)
- mỗi ô: trục x = lag (1-5 ngày), trục y = hệ số (đổi sang basis point,
x10000, cùng đơn vị Tetlock dùng để báo cáo), thanh sai số = 1,96 x SE
Newey-West (khoảng tin cậy 95%). Điểm có ý nghĩa (p<0,05, khoảng tin cậy
không chứa 0) tô đậm/đổi màu khác điểm không có ý nghĩa.

Import lại add_regression_features/fit_with_lag_sum_test từ file gốc
(Common/vnindex_daily_predictive_regression.py).

Output: data_News/sentiment_lag_coefficients{output_suffix}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from News_Vnindex.Common.vnindex_daily_predictive_regression import (
    N_LAGS,
    OUTPUT_DIR,
    SENTIMENT_COLUMN,
    TARGET_COLUMNS,
    add_regression_features,
    fit_with_lag_sum_test,
)


# Màu trung tính (không ý nghĩa) vs màu nhấn (có ý nghĩa, p<0.05) - 1 cặp
# diverging đơn giản, không dùng màu tùy tiện.
COLOR_NOT_SIGNIFICANT = "#94a3b8"  # xám ánh xanh
COLOR_SIGNIFICANT = "#dc2626"  # đỏ
COLOR_ZERO_LINE = "#334155"


def run_sentiment_lag_plot(paths_by_method: dict[str, Path], output_suffix: str = "") -> None:
    """1 hàng / method trong ``paths_by_method``, 1 cột / target.
    Output: data_News/sentiment_lag_coefficients{output_suffix}.png"""
    output_path = OUTPUT_DIR / f"sentiment_lag_coefficients{output_suffix}.png"

    method_names = list(paths_by_method.keys())
    n_rows, n_cols = len(method_names), len(TARGET_COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)

    for row_index, method_name in enumerate(method_names):
        merged_df = pd.read_parquet(paths_by_method[method_name])
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Đã lưu:", output_path)

