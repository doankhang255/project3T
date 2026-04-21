from pathlib import Path
from typing import Dict

import pandas as pd


RAW_NEWS_PARQUET_PATH = ""
WITH_KEYWORD_OUTPUT_PATH = "raw_news_with_keyword.parquet"
WITHOUT_KEYWORD_OUTPUT_PATH = "raw_news_without_keyword.parquet"


REQUIRED_COLUMNS = [
    "link",
    "domain",
    "category",
    "title",
    "description",
    "keywords",
    "author",
    "publication_date",
]

TEXT_COLUMNS = ["domain", "category", "title", "description", "author"]
NULL_LIKE_VALUES = {"", "null", "nan", "none"}


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def load_raw_news_parquet(parquet_path: str) -> pd.DataFrame:
    if not parquet_path.strip():
        raise ValueError("Set RAW_NEWS_PARQUET_PATH before running raw_news.py.")

    source_path = Path(parquet_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw news parquet not found: {parquet_path}")

    df = pd.read_parquet(source_path)
    df.columns = [column.strip().lower() for column in df.columns]
    validate_required_columns(df)
    return df


def parse_keywords_field(raw: object) -> object:
    if pd.isna(raw):
        return None

    text = str(raw).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return None

    return text


def clean_raw_news(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_df = df.copy()

    for column in TEXT_COLUMNS:
        cleaned_df[column] = cleaned_df[column].fillna("").astype(str).str.strip()

    cleaned_df["keywords"] = cleaned_df["keywords"].apply(parse_keywords_field)
    cleaned_df["publication_date"] = pd.to_datetime(cleaned_df["publication_date"], errors="coerce")
    duplicate_mask = cleaned_df.duplicated(subset=["link"], keep="first")
    removed_duplicates_df = cleaned_df[duplicate_mask].copy().reset_index(drop=True)
    cleaned_df = cleaned_df[~duplicate_mask].copy().reset_index(drop=True)
    return cleaned_df, removed_duplicates_df


def summarize_year_validation(df: pd.DataFrame) -> Dict[str, int]:
    if "year" not in df.columns:
        return {
            "match_rows": 0,
            "mismatch_rows": 0,
            "missing_rows": len(df),
        }

    publication_year = df["publication_date"].dt.year
    provided_year = pd.to_numeric(df["year"], errors="coerce")

    comparable_mask = publication_year.notna() & provided_year.notna()
    match_mask = comparable_mask & publication_year.eq(provided_year)
    mismatch_mask = comparable_mask & publication_year.ne(provided_year)
    missing_mask = ~comparable_mask

    return {
        "match_rows": int(match_mask.sum()),
        "mismatch_rows": int(mismatch_mask.sum()),
        "missing_rows": int(missing_mask.sum()),
    }


def summarize_keyword_presence(df: pd.DataFrame) -> Dict[str, int]:
    has_keyword_mask = df["keywords"].notna()
    return {
        "rows_with_keyword": int(has_keyword_mask.sum()),
        "rows_without_keyword": int((~has_keyword_mask).sum()),
    }


def split_news_by_keyword(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_keyword_mask = df["keywords"].notna()
    with_keyword_df = df[has_keyword_mask].copy()
    without_keyword_df = df[~has_keyword_mask].copy()
    return with_keyword_df, without_keyword_df


def main() -> None:
    raw_df = load_raw_news_parquet(RAW_NEWS_PARQUET_PATH)
    cleaned_df, removed_duplicates_df = clean_raw_news(raw_df)

    year_stats = summarize_year_validation(cleaned_df)
    keyword_stats = summarize_keyword_presence(cleaned_df)
    with_keyword_df, without_keyword_df = split_news_by_keyword(cleaned_df)

    with_keyword_df.to_parquet(WITH_KEYWORD_OUTPUT_PATH, index=False)
    without_keyword_df.to_parquet(WITHOUT_KEYWORD_OUTPUT_PATH, index=False)

    print("Done.")
    print("Total raw news rows:", len(cleaned_df))
    print("Year match rows:", year_stats["match_rows"])
    print("Year mismatch rows:", year_stats["mismatch_rows"])
    print("Year rows missing publication_date/year:", year_stats["missing_rows"])
    print("Rows with keyword:", keyword_stats["rows_with_keyword"])
    print("Rows without keyword:", keyword_stats["rows_without_keyword"])
    print("Output with keyword:", WITH_KEYWORD_OUTPUT_PATH)
    print("Output without keyword:", WITHOUT_KEYWORD_OUTPUT_PATH)
    print("Duplicate rows removed:", len(removed_duplicates_df))
    if not removed_duplicates_df.empty:
        print("Removed duplicate rows:")
        print(removed_duplicates_df.to_string(index=False))


if __name__ == "__main__":
    main()
