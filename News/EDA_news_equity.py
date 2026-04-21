import ast
from pathlib import Path
from typing import List

import pandas as pd
import re


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    text = normalize_text(text)
    return sum(1 for keyword in keywords if keyword in text)


# =========================
# 0. USER CONFIG
# =========================
EQUITY_NEWS_PATH = ""
EXPLODED_OUTPUT_PATH = "target_symbol_news_exploded.csv"
DAY_SENTIMENT_OUTPUT_PATH = "target_symbol_day_sentiment.csv"


# =========================
# 1. CONSTANTS
# =========================
REQUIRED_COLUMNS = ["link", "title", "description", "keywords", "publication_date"]
TEXT_COLUMNS = ["domain", "category", "title", "description", "keywords", "author"]

POSITIVE_KEYWORDS = [
    "tang tran",
    "but pha",
    "lai",
    "loi nhuan tang",
    "co tuc",
    "mua vao",
    "nang room",
    "niem yet",
    "thuong co phieu",
    "tang truong",
    "vuot ke hoach",
]

NEGATIVE_KEYWORDS = [
    "giam san",
    "lo ",
    "lo luy ke",
    "bi dinh chi",
    "huy niem yet",
    "bi han che giao dich",
    "ban manh",
    "thua lo",
    "suy giam",
    "ap luc",
]


# =========================
# 2. HELPERS
# =========================
def validate_equity_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def load_equity_news(path: str) -> pd.DataFrame:
    if not path.strip():
        raise ValueError("Set EQUITY_NEWS_PATH before running EDA_news_equity.py.")

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Equity news file not found: {path}")

    if source_path.suffix.lower() == ".csv":
        return pd.read_csv(source_path)
    if source_path.suffix.lower() == ".parquet":
        return pd.read_parquet(source_path)

    raise ValueError("EQUITY_NEWS_PATH must point to a .csv or .parquet file.")


def parse_saved_symbol_list(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(item).strip().upper() for item in raw if str(item).strip()]

    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "null"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip().upper() for item in parsed if str(item).strip()]
    except Exception:
        pass

    return [text.upper()]


def prepare_equity_news(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = df.copy()
    validate_equity_columns(prepared_df)

    for column in TEXT_COLUMNS:
        if column in prepared_df.columns:
            prepared_df[column] = prepared_df[column].fillna("").astype(str).str.strip()

    prepared_df["publication_date"] = pd.to_datetime(prepared_df["publication_date"], errors="coerce")

    if "pub_date" not in prepared_df.columns:
        prepared_df["pub_date"] = prepared_df["publication_date"].dt.date
    if "pub_year" not in prepared_df.columns:
        prepared_df["pub_year"] = prepared_df["publication_date"].dt.year

    prepared_df["title_clean"] = prepared_df["title"].apply(normalize_text)
    prepared_df["description_clean"] = prepared_df["description"].apply(normalize_text)
    prepared_df["keywords_clean"] = prepared_df["keywords"].apply(normalize_text)

    if "matched_symbols" in prepared_df.columns:
        prepared_df["matched_symbols"] = prepared_df["matched_symbols"].apply(parse_saved_symbol_list)
    elif "symbol" in prepared_df.columns:
        prepared_df["matched_symbols"] = prepared_df["symbol"].apply(parse_saved_symbol_list)
    else:
        raise ValueError("Equity file must contain either 'matched_symbols' or 'symbol'.")

    prepared_df["n_matched_symbols"] = prepared_df["matched_symbols"].apply(len)
    return prepared_df


def baseline_sentiment_score(row: pd.Series) -> int:
    positive_hits = 0
    negative_hits = 0

    for column in ["title_clean", "description_clean", "keywords_clean"]:
        positive_hits += count_keyword_hits(row[column], POSITIVE_KEYWORDS)
        negative_hits += count_keyword_hits(row[column], NEGATIVE_KEYWORDS)

    if positive_hits > negative_hits:
        return 1
    if negative_hits > positive_hits:
        return -1
    return 0


def explode_equity_news(df: pd.DataFrame) -> pd.DataFrame:
    exploded_df = df[df["n_matched_symbols"] > 0].copy().explode("matched_symbols")
    exploded_df["symbol"] = exploded_df["matched_symbols"].astype(str).str.strip().str.upper()
    exploded_df = exploded_df.drop(columns=["matched_symbols"])
    exploded_df = exploded_df[exploded_df["symbol"] != ""].reset_index(drop=True)
    return exploded_df


def aggregate_symbol_day_sentiment(exploded_df: pd.DataFrame) -> pd.DataFrame:
    return (
        exploded_df.groupby(["symbol", "pub_date"], as_index=False)
        .agg(
            news_count=("link", "count"),
            sentiment_score=("baseline_sentiment", "mean"),
        )
    )


def main() -> None:
    equity_df = load_equity_news(EQUITY_NEWS_PATH)
    prepared_equity_df = prepare_equity_news(equity_df)

    exploded_df = explode_equity_news(prepared_equity_df)
    exploded_df["baseline_sentiment"] = exploded_df.apply(baseline_sentiment_score, axis=1)

    symbol_day_sentiment = aggregate_symbol_day_sentiment(exploded_df)

    exploded_df.to_csv(EXPLODED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    symbol_day_sentiment.to_csv(DAY_SENTIMENT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Equity news rows:", len(prepared_equity_df))
    print("Exploded equity rows:", len(exploded_df))
    print("Symbol-day sentiment rows:", len(symbol_day_sentiment))


if __name__ == "__main__":
    main()
