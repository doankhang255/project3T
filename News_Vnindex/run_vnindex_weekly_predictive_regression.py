from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_merged.parquet"
ABNORMAL_INPUT_PATH = (PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return.parquet")
OUTPUT_PARQUET_PATH = (PROJECT_ROOT / "data_News" / "vnindex_weekly_predictive_regression.parquet")
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_predictive_regression.csv"

PREDICTOR_COLUMNS = [
    "sentiment_index_z",
    "return_lag_1w",
    "volatility_12w",
    "log_article_count",
]

TARGET_COLUMNS = [
    "future_ret_1w",
    "future_ret_4w",
    "future_abnormal_rolling_ret_1w",
    "future_abnormal_rolling_ret_4w",
    "future_abnormal_ar1_ret_1w",
    "future_abnormal_ar1_ret_4w",
]

HAC_LAGS_BY_TARGET = {
    "future_ret_1w": 1,
    "future_ret_4w": 4,
    "future_abnormal_rolling_ret_1w": 1,
    "future_abnormal_rolling_ret_4w": 4,
    "future_abnormal_ar1_ret_1w": 1,
    "future_abnormal_ar1_ret_4w": 4,
}


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, max_lag: int,) -> np.ndarray:
    n_obs = x.shape[0]
    x_residual = x * residuals[:, None]
    s_matrix = x_residual.T @ x_residual

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = x_residual[lag:].T @ x_residual[:-lag]
        s_matrix = s_matrix + weight * (gamma + gamma.T)

    x_tx_inverse = np.linalg.pinv(x.T @ x)
    covariance = x_tx_inverse @ s_matrix @ x_tx_inverse
    return covariance * n_obs / max(n_obs - x.shape[1], 1)


def fit_predictive_regression(df: pd.DataFrame, target_column: str, predictor_columns: list[str], hac_lags: int,) -> pd.DataFrame:

    model_df = df[[target_column] + predictor_columns].dropna().copy()
    if len(model_df) <= len(predictor_columns) + 2:
        return pd.DataFrame()

    y = model_df[target_column].to_numpy(dtype="float64")
    x_without_constant = model_df[predictor_columns].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(model_df)), x_without_constant])
    variable_names = ["const"] + predictor_columns

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted_values = x @ coefficients
    residuals = y - fitted_values

    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss != 0 else np.nan
    n_obs = len(model_df)
    n_params = x.shape[1]
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params)
        if n_obs > n_params and not pd.isna(r_squared)
        else np.nan
    )

    covariance = newey_west_covariance(x, residuals, max_lag=hac_lags)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=max(n_obs - n_params, 1))

    rows = []
    for variable_name, coefficient, standard_error, t_stat, p_value in zip(
        variable_names,
        coefficients,
        standard_errors,
        t_stats,
        p_values,
    ):
        rows.append(
            {
                "target_variable": target_column,
                "predictor_variable": variable_name,
                "coefficient": coefficient,
                "std_error_newey_west": standard_error,
                "t_stat": t_stat,
                "p_value": p_value,
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "observation_count": n_obs,
                "hac_lags": hac_lags,
            }
        )

    return pd.DataFrame(rows)


def build_regression_results(df: pd.DataFrame) -> pd.DataFrame:
    existing_predictors = [column for column in PREDICTOR_COLUMNS if column in df.columns
                           ]
    result_frames = []

    for target_column in TARGET_COLUMNS:
        if target_column not in df.columns:
            continue

        result = fit_predictive_regression(
            df,
            target_column=target_column,
            predictor_columns=existing_predictors,
            hac_lags=HAC_LAGS_BY_TARGET.get(target_column, 1),
        )
        if not result.empty:
            result_frames.append(result)

    if not result_frames:
        return pd.DataFrame()

    return pd.concat(result_frames, ignore_index=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = ABNORMAL_INPUT_PATH if ABNORMAL_INPUT_PATH.exists() else BASE_INPUT_PATH
    merged_df = pd.read_parquet(input_path)
    regression_result = build_regression_results(merged_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    regression_result.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    regression_result.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", input_path)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Input rows:", len(merged_df))
    print("Regression result rows:", len(regression_result))
    print(regression_result.to_string(index=False))


if __name__ == "__main__":
    main()
