"""Chẩn đoán: R² thấp (1,36%) là do đâu - từ biến kiểm soát hay do bản chất
bài toán khó đoán? Tách 3 mô hình LỒNG NHAU (nested), thêm dần từng nhóm
biến, xem R² tăng lên bao nhiêu ở mỗi bước - nhóm nào không đóng góp gì thì
lộ ra ngay:

    Mô hình A (mùa vụ):        const + dow_* + near_tet + volatility_20d
    Mô hình B (A + lịch sử):   + 5 lag chính target + 1 lag volume
    Mô hình C (B + sentiment): + 5 lag sentiment  (= mô hình đầy đủ đang dùng)

Cũng làm F-test kiểm định "thêm 5 lag sentiment vào B có làm mô hình tốt lên
CÓ Ý NGHĨA không" (kiểm định đồng thời cả 5 hệ số cùng lúc, khác kiểm định
tổng lag2-5 đã làm trước đó là kiểm định RIÊNG lag2-5, không phải cả 5 lag).

LƯU Ý: F-test kinh điển này giả định phần dư đồng nhất/không tự tương quan
(không phải Newey-West) - chỉ dùng để CHẨN ĐOÁN nhanh xem nhóm biến có đóng
góp gì không, không thay thế cho p-value Newey-West đã dùng để KẾT LUẬN
chính thức ở các phần trước.

Chạy trên 1 tổ hợp đại diện (Cach1_PMI / abnormal_return_ar1_1d).

Output: data_News/model_r_squared_diagnosis.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnindex_daily_predictive_regression import (  # noqa: E402
    MERGED_DATA_PATHS_BY_METHOD,
    N_LAGS,
    SENTIMENT_COLUMN,
    add_regression_features,
)

REPRESENTATIVE_METHOD = "Cach1_PMI"
REPRESENTATIVE_TARGET = "abnormal_return_ar1_1d"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "model_r_squared_diagnosis.png"


def fit_r_squared(model_df: pd.DataFrame, target_column: str, predictor_columns: list[str]) -> tuple[float, float, int, int]:
    """OLS thường (không Newey-West, chỉ để so R² và RSS giữa các mô hình
    lồng nhau). Trả về (r_squared, rss, n_obs, n_params)."""
    if predictor_columns:
        x = np.column_stack([np.ones(len(model_df)), model_df[predictor_columns].to_numpy(dtype="float64")])
    else:
        x = np.ones((len(model_df), 1))
    y = model_df[target_column].to_numpy(dtype="float64")

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss
    return r_squared, rss, len(model_df), x.shape[1]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    merged_df = pd.read_parquet(MERGED_DATA_PATHS_BY_METHOD[REPRESENTATIVE_METHOD])
    featured_df, sentiment_lag_columns, all_predictor_columns = add_regression_features(
        merged_df, REPRESENTATIVE_TARGET
    )
    model_df = featured_df[[REPRESENTATIVE_TARGET] + all_predictor_columns].dropna()

    target_lag_columns = [f"{REPRESENTATIVE_TARGET}_lag{lag}" for lag in range(1, N_LAGS + 1)]
    seasonal_columns = [
        column for column in all_predictor_columns if column not in sentiment_lag_columns + target_lag_columns
        and not column.startswith("log_vol_total")
    ]
    volume_columns = [column for column in all_predictor_columns if column.startswith("log_vol_total")]

    model_a_columns = seasonal_columns
    model_b_columns = seasonal_columns + target_lag_columns + volume_columns
    model_c_columns = model_b_columns + sentiment_lag_columns

    r2_a, rss_a, n_obs, k_a = fit_r_squared(model_df, REPRESENTATIVE_TARGET, model_a_columns)
    r2_b, rss_b, _, k_b = fit_r_squared(model_df, REPRESENTATIVE_TARGET, model_b_columns)
    r2_c, rss_c, _, k_c = fit_r_squared(model_df, REPRESENTATIVE_TARGET, model_c_columns)

    print(f"Method/target đại diện: {REPRESENTATIVE_METHOD} / {REPRESENTATIVE_TARGET}  (n={n_obs})")
    print()
    print(f"Mô hình A (chỉ mùa vụ: dow/near_tet/volatility, {len(model_a_columns)} biến):    R² = {r2_a:.5f}")
    print(f"Mô hình B (A + lịch sử return/volume, {len(model_b_columns)} biến):        R² = {r2_b:.5f}   (+{r2_b-r2_a:.5f} so với A)")
    print(f"Mô hình C (B + sentiment, {len(model_c_columns)} biến = mô hình đầy đủ):  R² = {r2_c:.5f}   (+{r2_c-r2_b:.5f} so với B)")

    # F-test: B (restricted) vs C (full) - 5 sentiment lags cùng lúc
    q = k_c - k_b
    f_stat = ((rss_b - rss_c) / q) / (rss_c / (n_obs - k_c))
    f_p_value = float(stats.f.sf(f_stat, q, n_obs - k_c))
    print()
    print(f"F-test (thêm 5 lag sentiment vào mô hình B): F({q},{n_obs-k_c}) = {f_stat:.3f}, p = {f_p_value:.4f}")
    print("(OLS thường, không Newey-West - chỉ để chẩn đoán nhanh, không phải kết luận chính thức)")

    # --- Vẽ ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    models = ["A: mùa vụ", "B: + lịch sử\nreturn/volume", "C: + sentiment\n(đầy đủ)"]
    r2_values = [r2_a, r2_b, r2_c]
    colors = ["#94a3b8", "#0ea5e9", "#dc2626"]
    ax_left.bar(models, [value * 100 for value in r2_values], color=colors)
    for index, value in enumerate(r2_values):
        ax_left.text(index, value * 100 + 0.03, f"{value*100:.2f}%", ha="center", fontsize=9)
    ax_left.set_ylabel("R² (%)", fontsize=9)
    ax_left.set_title("R² tích lũy theo từng nhóm biến thêm vào", fontsize=10)
    ax_left.tick_params(labelsize=8)

    increments = [r2_a, r2_b - r2_a, r2_c - r2_b]
    increment_labels = ["A\n(mùa vụ)", "B-A\n(lịch sử)", "C-B\n(sentiment)"]
    ax_right.bar(increment_labels, [value * 100 for value in increments], color=colors)
    for index, value in enumerate(increments):
        ax_right.text(index, value * 100 + (0.02 if value >= 0 else -0.05), f"{value*100:.3f}%", ha="center", fontsize=9)
    ax_right.axhline(0, color="#334155", linewidth=0.8)
    ax_right.set_ylabel("Đóng góp R² riêng (%)", fontsize=9)
    ax_right.set_title(
        f"Đóng góp riêng từng nhóm\nF-test sentiment: p={f_p_value:.3f}", fontsize=10
    )
    ax_right.tick_params(labelsize=8)

    fig.suptitle(f"Chẩn đoán R² - {REPRESENTATIVE_METHOD} / {REPRESENTATIVE_TARGET} (n={n_obs})", fontsize=11)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print()
    print("Đã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
