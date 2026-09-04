from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.model.common import DATA_DIR


MODEL_METRICS_FILES = {
    "logistic_regression": DATA_DIR / "logistic_regression_metrics.csv",
    "naive_bayes": DATA_DIR / "naive_bayes_metrics.csv",
    "svm": DATA_DIR / "svm_metrics.csv",
    "random_forest": DATA_DIR / "random_forest_metrics.csv",
}

OUTPUT_COMPARISON_PATH = DATA_DIR / "traditional_ml_model_comparison.csv"


def load_model_metrics(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file for {model_name!r} not found: {path}. "
            "Run the full pipeline first: "
            "python News/Build_sentiment_label/Traditional_ML/run_pipeline.py"
        )
    df = pd.read_csv(path)
    df.insert(0, "model", model_name)
    return df


def build_comparison_table(
    metrics_files: dict[str, Path] = MODEL_METRICS_FILES,
) -> pd.DataFrame:
    frames = [
        load_model_metrics(model_name, path)
        for model_name, path in metrics_files.items()
    ]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    comparison_df = build_comparison_table()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    print("Per-class + overall metrics, all models:")
    print(comparison_df.to_string(index=False))

    overall_df = comparison_df.loc[
        comparison_df["metric_scope"].eq("overall"),
        ["model", "f1", "accuracy"],
    ].rename(columns={"f1": "macro_f1"})
    overall_df = overall_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    print("\nOverall summary, ranked by macro F1:")
    print(overall_df.to_string(index=False))

    print("\nOutput comparison csv:", OUTPUT_COMPARISON_PATH)


if __name__ == "__main__":
    main()
