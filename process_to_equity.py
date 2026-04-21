import ast
from pathlib import Path
import re
from typing import List, Set

import pandas as pd

INPUT_PATH = ""
OUTPUT_PATH = "equity.csv"
NOT_EQUITY_OUTPUT_PATH = "not_equity.csv"
MANUAL_TICKERS: List[str] = []
CATEGORY_KEEP_KEYWORDS = [
    "bất động sản",
    "kinh doanh",
    "thông tin doanh nghiệp",
    "thông tin đầu tư",
    "tài chính",
    "ngân hàng",
]
NULL_LIKE_VALUES = {"", "null", "nan", "none"}
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
def load_input_data(path: str) -> pd.DataFrame:
    if not path.strip():
        raise ValueError("Set INPUT_PATH before running process_to_equity.py.")

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if source_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_path)
    elif source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        raise ValueError("INPUT_PATH must point to a .csv or .parquet file.")

    df.columns = [column.strip().lower() for column in df.columns]
    return df


def get_keyword_column(df: pd.DataFrame) -> str | None:
    if "keywords" in df.columns:
        return "keywords"
    if "keyword" in df.columns:
        return "keyword"
    return None


def get_description_column(df: pd.DataFrame) -> str | None:
    if "description" in df.columns:
        return "description"
    return None


def build_ticker_set(tickers: List[str]) -> Set[str]:
    ticker_set = {ticker.strip() for ticker in tickers if ticker.strip()}
    if not ticker_set:
        raise ValueError("MANUAL_TICKERS is empty. Add ticker codes before running.")
    return ticker_set


def parse_keyword_items(raw: object) -> List[str]:
    if pd.isna(raw):
        return []

    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]

    text = str(raw).strip()
    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return []

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return [str(key).strip() for key in parsed.keys() if str(key).strip()]
        except Exception:
            pass

    return [item.strip() for item in text.split(",") if item.strip()]


def extract_word_tokens(text: object) -> Set[str]:
    if pd.isna(text):
        return set()

    text = str(text).strip()
    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return set()

    return {token for token in TOKEN_PATTERN.findall(text) if token}


def match_tickers_in_keywords(raw: object, ticker_set: Set[str]) -> List[str]:
    keyword_items = parse_keyword_items(raw)
    matched_tickers = set()

    for item in keyword_items:
        if item in ticker_set:
            matched_tickers.add(item)
            continue

        item_tokens = extract_word_tokens(item)
        matched_tickers.update(item_tokens.intersection(ticker_set))

    return sorted(matched_tickers)


def match_tickers_in_description(raw: object, ticker_set: Set[str]) -> List[str]:
    description_tokens = extract_word_tokens(raw)
    return sorted(description_tokens.intersection(ticker_set))


def has_meaningful_text(raw: object) -> bool:
    if pd.isna(raw):
        return False

    text = str(raw).strip()
    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return False

    return True


def category_is_equity_relevant(raw: object, keep_keywords: List[str]) -> bool:
    if pd.isna(raw):
        return False

    text = str(raw).strip().lower()
    if text == "" or text in NULL_LIKE_VALUES:
        return False

    return any(keyword in text for keyword in keep_keywords)


def main() -> None:
    df = load_input_data(INPUT_PATH)
    ticker_set = build_ticker_set(MANUAL_TICKERS)

    category_mask = df["category"].apply(
        lambda value: category_is_equity_relevant(value, CATEGORY_KEEP_KEYWORDS)
    )
    removed_category_rows = int((~category_mask).sum())
    df = df[category_mask].copy().reset_index(drop=True)

    keyword_column = get_keyword_column(df)
    description_column = get_description_column(df)
    if keyword_column is None and description_column is None:
        raise ValueError("Input file must contain at least one of: 'keyword', 'keywords', 'description'.")

    if keyword_column is not None:
        keyword_mask = df[keyword_column].apply(has_meaningful_text)
        df["matched_tickers_keyword"] = df[keyword_column].apply(
            lambda value: match_tickers_in_keywords(value, ticker_set)
        )
    else:
        keyword_mask = pd.Series(False, index=df.index)
        df["matched_tickers_keyword"] = [[] for _ in range(len(df))]

    if description_column is not None:
        description_mask = df[description_column].apply(has_meaningful_text)
        df["matched_tickers_description"] = df[description_column].apply(
            lambda value: match_tickers_in_description(value, ticker_set)
        )
    else:
        description_mask = pd.Series(False, index=df.index)
        df["matched_tickers_description"] = [[] for _ in range(len(df))]

    removable_mask = (~keyword_mask) & (~description_mask)
    removed_rows = int(removable_mask.sum())
    df = df[~removable_mask].copy().reset_index(drop=True)

    df["matched_tickers"] = df.apply(
        lambda row: sorted(
            set(row["matched_tickers_keyword"]).union(set(row["matched_tickers_description"]))
        ),
        axis=1,
    )
    df["is_equity"] = df["matched_tickers"].apply(len) == 1

    equity_df = df[df["is_equity"]].copy()
    not_equity_df = df[~df["is_equity"]].copy()
    equity_df["matched_tickers_keyword"] = equity_df["matched_tickers_keyword"].apply(lambda values: ",".join(values))
    equity_df["matched_tickers_description"] = equity_df["matched_tickers_description"].apply(lambda values: ",".join(values))
    equity_df["matched_tickers"] = equity_df["matched_tickers"].apply(lambda values: ",".join(values))
    not_equity_df["matched_tickers_keyword"] = not_equity_df["matched_tickers_keyword"].apply(lambda values: ",".join(values))
    not_equity_df["matched_tickers_description"] = not_equity_df["matched_tickers_description"].apply(lambda values: ",".join(values))
    not_equity_df["matched_tickers"] = not_equity_df["matched_tickers"].apply(lambda values: ",".join(values))

    equity_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    not_equity_df.to_csv(NOT_EQUITY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Input rows:", len(df) + removed_rows + removed_category_rows)
    print("Removed rows with invalid category:", removed_category_rows)
    print("Removed rows with no keyword and no description:", removed_rows)
    print("Equity rows:", int(df["is_equity"].sum()))
    print("Non-equity rows:", int((~df["is_equity"]).sum()))
    print("Equity output file:", OUTPUT_PATH)
    print("Not-equity output file:", NOT_EQUITY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
