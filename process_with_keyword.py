import ast
from pathlib import Path
import re
from typing import List, Set

import pandas as pd


INPUT_PATH = "raw_news_with_keyword.parquet"
OUTPUT_PATH = "equity_with_keyword.csv"

# Add your tickers here.
MANUAL_TICKERS: List[str] = []

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def load_input_data(path: str) -> pd.DataFrame:
    if not path.strip():
        raise ValueError("Set INPUT_PATH before running process_with_keyword.py.")

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


def get_keyword_column(df: pd.DataFrame) -> str:
    if "keywords" in df.columns:
        return "keywords"
    if "keyword" in df.columns:
        return "keyword"
    raise ValueError("Input file must contain either 'keyword' or 'keywords' column.")


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
    if text == "" or text.lower() in {"null", "nan", "none"}:
        return []

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return [str(key).strip() for key in parsed.keys() if str(key).strip()]
        except Exception:
            pass

    return [item.strip() for item in text.split(",") if item.strip()]


def extract_word_tokens(text: str) -> Set[str]:
    return {token for token in TOKEN_PATTERN.findall(str(text)) if token}


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


def main() -> None:
    df = load_input_data(INPUT_PATH)
    keyword_column = get_keyword_column(df)
    ticker_set = build_ticker_set(MANUAL_TICKERS)

    df["matched_tickers"] = df[keyword_column].apply(lambda value: match_tickers_in_keywords(value, ticker_set))
    df["n_matched_tickers"] = df["matched_tickers"].apply(len)
    df["is_equity"] = df["n_matched_tickers"] == 1

    equity_df = df[df["is_equity"]].copy()
    equity_df["matched_tickers"] = equity_df["matched_tickers"].apply(lambda values: ",".join(values))

    equity_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Input rows:", len(df))
    print("Equity rows:", int(df["is_equity"].sum()))
    print("Non-equity rows:", int((~df["is_equity"]).sum()))
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
