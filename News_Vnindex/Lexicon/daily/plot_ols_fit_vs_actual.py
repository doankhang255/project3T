"""Biểu đồ đường: giá trị THỰC (abnormal return) vs giá trị MÔ HÌNH OLS DỰ
BÁO (fitted = X @ coefficients) theo thời gian - để nhìn trực quan "OLS học
được bao nhiêu" và sai số (residual = actual - fitted) lớn cỡ nào so với
chính lợi suất thực.

2 phần:
    1. Đường thực tế vs đường dự báo, chồng lên nhau (1 năm gần nhất, để còn
       đọc được - vẽ hết ~3350 ngày sẽ đặc kín không thấy gì).
    2. Đường sai số (residual = actual - fitted) cùng khung thời gian, để
       thấy sai số dao động quanh 0 với biên độ so với đường thực tế ở trên.

Kèm bảng số: RMSE, MAE, R² - đo mức "học được" bằng số thay vì chỉ nhìn hình.

Chạy trên 1 tổ hợp đại diện (Cach1_PMI / abnormal_return_ar1_1d).

Import lại add_regression_features từ file gốc (không viết lại).

Output: data_News/ols_fit_vs_actual.png
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
    add_regression_features,
)

REPRESENTATIVE_METHOD = "Cach1_PMI"
REPRESENTATIVE_TARGET = "abnormal_return_ar1_1d"
ZOOM_TRADING_DAYS = 252  # ~1 nam gan nhat, de con doc duoc hinh
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "ols_fit_vs_actual.png"

COLOR_ACTUAL = "#1d4ed8"
COLOR_FITTED = "#ea580c"
COLOR_RESIDUAL = "#64748b"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    merged_df = pd.read_parquet(MERGED_DATA_PATHS_BY_METHOD[REPRESENTATIVE_METHOD])
    featured_df, _sentiment_lag_columns, predictor_columns = add_regression_features(
        merged_df, REPRESENTATIVE_TARGET
    )
    model_df = featured_df[["date", REPRESENTATIVE_TARGET] + predictor_columns].dropna().copy()
    model_df = model_df.sort_values("date").reset_index(drop=True)

    y = model_df[REPRESENTATIVE_TARGET].to_numpy(dtype="float64")
    x_without_constant = model_df[predictor_columns].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(model_df)), x_without_constant])

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficients
    residuals = y - fitted

    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    mean_abs_actual = float(np.mean(np.abs(y)))

    print(f"Method/target đại diện: {REPRESENTATIVE_METHOD} / {REPRESENTATIVE_TARGET}")
    print(f"Số quan sát: {len(model_df)}")
    print(f"R-squared:              {r_squared:.5f}  ({r_squared*100:.3f}% biến thiên được giải thích)")
    print(f"RMSE (sai số):          {rmse*10000:.2f} basis point")
    print(f"MAE (sai số tuyệt đối): {mae*10000:.2f} basis point")
    print(f"Biên độ trung bình |return thực|: {mean_abs_actual*10000:.2f} basis point")
    print(f"=> Sai số trung bình (MAE) bằng {mae/mean_abs_actual*100:.1f}% biên độ return thực trung bình")

    model_df["fitted"] = fitted
    model_df["residual"] = residuals

    plot_df = model_df.tail(ZOOM_TRADING_DAYS)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1])

    ax_top.plot(
        plot_df["date"], plot_df[REPRESENTATIVE_TARGET] * 10000, color=COLOR_ACTUAL, linewidth=1.1, label="Thực tế"
    )
    ax_top.plot(
        plot_df["date"], plot_df["fitted"] * 10000, color=COLOR_FITTED, linewidth=1.6, label="OLS dự báo (fitted)"
    )
    ax_top.axhline(0, color="#334155", linewidth=0.8, linestyle="--")
    ax_top.set_ylabel("Abnormal return (basis point)", fontsize=9)
    ax_top.set_title(
        f"Thực tế vs OLS dự báo - {REPRESENTATIVE_METHOD} / {REPRESENTATIVE_TARGET}\n"
        f"({ZOOM_TRADING_DAYS} ngày giao dịch gần nhất - R²={r_squared:.4f}, "
        f"RMSE={rmse*10000:.1f}bp, MAE={mae*10000:.1f}bp)",
        fontsize=11,
    )
    ax_top.legend(loc="upper right", fontsize=9, frameon=False)
    ax_top.tick_params(labelsize=8)

    ax_bottom.plot(plot_df["date"], plot_df["residual"] * 10000, color=COLOR_RESIDUAL, linewidth=1.0)
    ax_bottom.axhline(0, color="#334155", linewidth=0.8, linestyle="--")
    ax_bottom.fill_between(plot_df["date"], plot_df["residual"] * 10000, 0, color=COLOR_RESIDUAL, alpha=0.15)
    ax_bottom.set_ylabel("Sai số\n(thực tế - fitted, bp)", fontsize=9)
    ax_bottom.set_xlabel("Ngày", fontsize=9)
    ax_bottom.tick_params(labelsize=8)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print()
    print("Đã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
