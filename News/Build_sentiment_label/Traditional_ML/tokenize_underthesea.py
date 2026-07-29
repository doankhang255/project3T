from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "Common"

INPUT_PATH = PROJECT_ROOT / "data_news" / "ground_truth_labeled.csv"
OUTPUT_CSV_PATH = SCRIPT_DIR / "data" / "ground_truth_labeled_tokenized.csv"
OUTPUT_PARQUET_PATH = SCRIPT_DIR / "data" / "ground_truth_labeled_tokenized.parquet"

TEXT_COLUMN = "content"
TOKENIZED_COLUMN = "Tokenize_content"
TOKENIZED_SENTENCES_COLUMN = "Tokenize_content_sentences"
TOKENIZED_TEXT_COLUMN = "Tokenize_content_text"
TOTAL_TOKENIZER_COLUMN = "total_tokenizer"


if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

try:
    from tokenize_underthesea import tokenize_vietnamese_text_by_sentence  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name == "underthesea":
        raise ModuleNotFoundError(
            "Missing dependency 'underthesea'. Install it in the active "
            "environment before running this script."
        ) from exc
    raise


def load_ground_truth(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input file is missing column: {TEXT_COLUMN!r}")

    return df


def tokenize_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TOKENIZED_SENTENCES_COLUMN] = out[TEXT_COLUMN].apply(
        tokenize_vietnamese_text_by_sentence
    )
    out[TOKENIZED_COLUMN] = out[TOKENIZED_SENTENCES_COLUMN].apply(
        lambda sentences: [token for sentence in sentences for token in sentence]
    )
    out[TOKENIZED_TEXT_COLUMN] = out[TOKENIZED_COLUMN].apply(" ".join)
    out[TOTAL_TOKENIZER_COLUMN] = out[TOKENIZED_COLUMN].apply(len)
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_ground_truth(INPUT_PATH)
    tokenized_df = tokenize_ground_truth(df)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenized_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    tokenized_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print("Input:", INPUT_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Rows:", len(tokenized_df))
    print("Total tokens:", int(tokenized_df[TOTAL_TOKENIZER_COLUMN].sum()))
    print(
        tokenized_df[
            ["id", "sentiment", "title", TOKENIZED_TEXT_COLUMN, TOTAL_TOKENIZER_COLUMN]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
