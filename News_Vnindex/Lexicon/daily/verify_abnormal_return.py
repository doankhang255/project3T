"""Kiểm chứng abnormal_return được tính đúng - 2 phép kiểm tra, vẽ 2 biểu đồ.

1. Identity check: theo định nghĩa abnormal = actual - expected, nên
   expected + abnormal phải khớp TUYỆT ĐỐI với daily_return gốc. Vẽ scatter
   (x=daily_return thật, y=expected+abnormal tính lại) - phải nằm đúng 1
   đường chéo 45 độ, sai lệch chỉ do làm tròn số thực (float), gần 0 tuyệt
   đối. Nếu lệch khỏi đường chéo -> có bug ở đâu đó trong công thức.

2. Autocorrelation check: abnormal_return_ar1 là phần dư của hồi quy
   return_t ~ a + b*return_{t-1} - nếu tính đúng, phần dư phải gần như
   KHÔNG còn tương quan với return_lag_1 nữa (đã "vắt" hết vào expected).
   So sánh ACF (tự tương quan, lag 1-10) của daily_return gốc (sẽ thấy
   tương quan ở lag 1) với ACF của abnormal_return_ar1 (phải giảm mạnh về
   gần 0 ở lag 1) - nếu abnormal_return_ar1 vẫn còn tương quan cao gần
   bằng daily_return gốc thì việc "khử" không hiệu quả.

Output: data_News/verify_abnormal_return.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_abnormal_return_pmi.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data_News" / "verify_abnormal_return.png"

MAX_LAG = 10


def autocorrelation(series: pd.Series, max_lag: int) -> list[float]:
    """ACF thủ công: corrcoef(x_t, x_{t-lag}) cho từng lag - không dùng thư
    viện ngoài, minh bạch công thức."""
    clean = series.dropna().to_numpy(dtype="float64")
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= len(clean):
            result.append(np.nan)
            continue
        correlation = np.corrcoef(clean[lag:], clean[:-lag])[0, 1]
        result.append(correlation)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_parquet(
        INPUT_PATH,
        columns=["date", "daily_return", "expected_return_rolling", "abnormal_return_rolling_1d",
                 "expected_return_ar1", "abnormal_return_ar1_1d"],
    )
    df = df.dropna(subset=["daily_return"]).sort_values("date")

    # --- Kiểm tra 1: identity check ---
    reconstructed_rolling = df["expected_return_rolling"] + df["abnormal_return_rolling_1d"]
    reconstructed_ar1 = df["expected_return_ar1"] + df["abnormal_return_ar1_1d"]
    check_df = df.loc[df["expected_return_rolling"].notna()].copy()
    check_df["reconstructed_rolling"] = reconstructed_rolling
    max_diff_rolling = (check_df["reconstructed_rolling"] - check_df["daily_return"]).abs().max()

    check_ar1_df = df.loc[df["expected_return_ar1"].notna()].copy()
    check_ar1_df["reconstructed_ar1"] = reconstructed_ar1
    max_diff_ar1 = (check_ar1_df["reconstructed_ar1"] - check_ar1_df["daily_return"]).abs().max()

    print("=== Kiểm tra 1: identity check (expected + abnormal == daily_return?) ===")
    print(f"Rolling: sai lệch tối đa = {max_diff_rolling:.2e} (phải ~0, chỉ do làm tròn float)")
    print(f"AR(1):   sai lệch tối đa = {max_diff_ar1:.2e} (phải ~0, chỉ do làm tròn float)")

    # --- Kiểm tra 2: autocorrelation check ---
    acf_daily_return = autocorrelation(df["daily_return"], MAX_LAG)
    acf_abnormal_ar1 = autocorrelation(df["abnormal_return_ar1_1d"], MAX_LAG)

    print()
    print("=== Kiểm tra 2: tự tương quan (ACF) lag 1-10 ===")
    print(f"{'lag':>4} {'daily_return (gốc)':>20} {'abnormal_return_ar1':>22}")
    for lag_index in range(MAX_LAG):
        print(f"{lag_index+1:>4} {acf_daily_return[lag_index]:>20.4f} {acf_abnormal_ar1[lag_index]:>22.4f}")

    # --- Vẽ ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    sample = check_ar1_df.sample(min(2000, len(check_ar1_df)), random_state=0)
    ax_left.scatter(sample["daily_return"] * 10000, sample["reconstructed_ar1"] * 10000, s=6, alpha=0.4, color="#1d4ed8")
    lims = [
        min(sample["daily_return"].min(), sample["reconstructed_ar1"].min()) * 10000,
        max(sample["daily_return"].max(), sample["reconstructed_ar1"].max()) * 10000,
    ]
    ax_left.plot(lims, lims, color="#dc2626", linewidth=1.2, linestyle="--", label="Đường chéo 45° (khớp hoàn hảo)")
    ax_left.set_xlabel("daily_return gốc (bp)", fontsize=9)
    ax_left.set_ylabel("expected_return_ar1 + abnormal_return_ar1_1d (bp)", fontsize=9)
    ax_left.set_title(f"Kiểm tra 1: Identity check\nsai lệch tối đa = {max_diff_ar1:.1e}", fontsize=10)
    ax_left.legend(fontsize=8, frameon=False)
    ax_left.tick_params(labelsize=8)

    lags = list(range(1, MAX_LAG + 1))
    width = 0.38
    ax_right.bar([lag - width / 2 for lag in lags], acf_daily_return, width=width, color="#1d4ed8", label="daily_return (gốc)")
    ax_right.bar([lag + width / 2 for lag in lags], acf_abnormal_ar1, width=width, color="#ea580c", label="abnormal_return_ar1_1d")
    ax_right.axhline(0, color="#334155", linewidth=0.8)
    ax_right.set_xlabel("Lag (ngày)", fontsize=9)
    ax_right.set_ylabel("Tự tương quan (ACF)", fontsize=9)
    ax_right.set_title("Kiểm tra 2: ACF gốc vs sau khi khử\n(abnormal phải gần 0 hơn hẳn ở lag 1)", fontsize=10)
    ax_right.set_xticks(lags)
    ax_right.legend(fontsize=8, frameon=False)
    ax_right.tick_params(labelsize=8)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print()
    print("Đã lưu:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
