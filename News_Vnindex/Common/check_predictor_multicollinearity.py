"""
Dùng lại THẲNG `add_regression_features()` từ Common/vnindex_daily_predictive_regression.py
(không viết lại danh sách predictor) - đảm bảo đúng y hệt 21 biến đang dùng
thật trong hồi quy, không lệch nếu sau này predictor đổi.

2 phép đo:
    - Correlation matrix (Pearson, cặp đôi) - trực quan, dễ đọc nhưng chỉ bắt
      được cộng tuyến GIỮA 2 BIẾN, bỏ sót cộng tuyến bậc cao (VD X3 = X1+X2).
    - VIF (Variance Inflation Factor) - mỗi biến hồi quy trên TẤT CẢ biến còn
      lại, VIF = 1/(1-R²) - bắt được cả cộng tuyến bậc cao. Quy ước: VIF>5
      đáng chú ý, VIF>10 nghiêm trọng (dấu hiệu biến dư thừa).

Chạy trên 1 tổ hợp đại diện (method ĐẦU TIÊN của nhánh, target=abnormal_return_ar1_1d) - bộ
predictor sentiment/volume/dummy giống hệt ở mọi tổ hợp, chỉ khác target nên
không cần chạy cả 4 tổ hợp.

Output:
    data_News/predictor_correlation_heatmap{output_suffix}.png
    data_News/predictor_vif{output_suffix}.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from News_Vnindex.Common.vnindex_daily_predictive_regression import (
    OUTPUT_DIR,
    add_regression_features,
)

REPRESENTATIVE_TARGET = "abnormal_return_ar1_1d"


def compute_vif(predictor_df: pd.DataFrame) -> pd.DataFrame:
    """VIF cho từng cột: hồi quy cột đó trên tất cả cột còn lại, VIF = 1/(1-R²).
    R² cao (cột này gần như là tổ hợp tuyến tính của các cột khác) -> VIF cao."""
    columns = list(predictor_df.columns)
    x_all = predictor_df.to_numpy(dtype="float64")

    rows = []
    for target_index, target_name in enumerate(columns):
        other_indices = [index for index in range(len(columns)) if index != target_index]
        y = x_all[:, target_index]
        x_others = np.column_stack([np.ones(len(y)), x_all[:, other_indices]])

        coefficients, *_ = np.linalg.lstsq(x_others, y, rcond=None)
        fitted = x_others @ coefficients
        residuals = y - fitted

        rss = float(np.sum(residuals**2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - rss / tss if tss != 0 else 0.0
        vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else np.inf

        rows.append({"predictor": target_name, "r_squared_vs_others": r_squared, "vif": vif})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def run_multicollinearity_check(paths_by_method: dict[str, Path], output_suffix: str = "") -> None:
    """Method đại diện = method ĐẦU TIÊN của ``paths_by_method``. Output:
    data_News/predictor_correlation_heatmap{output_suffix}.png
    data_News/predictor_vif{output_suffix}.csv"""
    representative_method = next(iter(paths_by_method))
    heatmap_path = OUTPUT_DIR / f"predictor_correlation_heatmap{output_suffix}.png"
    vif_csv_path = OUTPUT_DIR / f"predictor_vif{output_suffix}.csv"

    merged_df = pd.read_parquet(paths_by_method[representative_method])
    featured_df, _sentiment_lag_columns, predictor_columns = add_regression_features(
        merged_df, REPRESENTATIVE_TARGET
    )
    predictor_df = featured_df[predictor_columns].dropna()
    print(f"Method/target đại diện: {representative_method} / {REPRESENTATIVE_TARGET}")
    print("Số predictor:", len(predictor_columns))
    print("Số dòng sau dropna:", len(predictor_df))

    # --- Correlation heatmap ---
    correlation_matrix = predictor_df.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        correlation_matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Hệ số tương quan Pearson"},
        annot=False,
        ax=ax,
    )
    ax.set_title(
        f"Tương quan giữa các predictor - {representative_method} / {REPRESENTATIVE_TARGET}",
        fontsize=12,
    )
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print("Đã lưu heatmap:", heatmap_path)

    # --- VIF ---
    vif_df = compute_vif(predictor_df)
    vif_df.to_csv(vif_csv_path, index=False, encoding="utf-8-sig")
    print("Đã lưu VIF:", vif_csv_path)
    print()
    print("Top 10 predictor có VIF cao nhất (VIF>10 = cộng tuyến nghiêm trọng, VIF>5 = đáng chú ý):")
    print(vif_df.head(10).to_string(index=False))

    print()
    high_vif = vif_df.loc[vif_df["vif"] > 10]
    if len(high_vif):
        print(f"CẢNH BÁO: {len(high_vif)} predictor có VIF > 10:")
        print(high_vif.to_string(index=False))
    else:
        print("Không có predictor nào VIF > 10.")

