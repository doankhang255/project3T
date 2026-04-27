from __future__ import annotations
import ast
import re
from numpy import iterable
import pandas as pd


from News.EDA_raw_news import add_description_word_count
from News.EDA_raw_news import calculate_description_word_count_percentiles
from News.EDA_raw_news import build_unique_value_counts
from News.EDA_raw_news import INPUT_PATH
from News.EDA_raw_news import load_input_data
from News.EDA_raw_news import NULL_LIKE_VALUES
from News.EDA_raw_news import TEXT_COLUMNS


def normalize_domain_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    normalize_domain = out.copy()
    domain = normalize_domain["domain"].astype("string").str.strip().str.lower()
    domain = domain.mask(domain.isna() | domain.fillna("").isin(NULL_LIKE_VALUES))
    domain = domain.str.replace(r"^https?://", "", regex=True)
    domain = domain.str.replace(r"^www\.", "", regex=True)
    domain = domain.str.split("/", n=1).str[0]
    domain = domain.str.split("?", n=1).str[0]
    domain = domain.str.split("#", n=1).str[0]
    domain = domain.str.strip()
    normalize_domain["domain"] = domain.mask(domain.fillna("").eq("")).astype("string")
    return out, normalize_domain


def parse_dict_like_keywords(raw: object) -> object:
    if pd.isna(raw):
        return pd.NA

    if isinstance(raw, dict):
        keys = [str(key).strip() for key in raw.keys() if str(key).strip()]
        return ", ".join(keys) if keys else pd.NA

    text = str(raw).strip()
    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return pd.NA
    if not (text.startswith("{") and text.endswith("}")):
        return text

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text

    if isinstance(parsed, dict):
        keys = [str(key).strip() for key in parsed.keys() if str(key).strip()]
        return ", ".join(keys) if keys else pd.NA
    return text


def is_dict_like_keywords(raw: object) -> bool:
    if pd.isna(raw):
        return False

    if isinstance(raw, dict):
        return True

    text = str(raw).strip()
    if text == "" or text.lower() in NULL_LIKE_VALUES:
        return False
    if not (text.startswith("{") and text.endswith("}")):
        return False

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return False
    return isinstance(parsed, dict)


def normalize_keywords_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["keywords"] = out["keywords"].apply(parse_dict_like_keywords).astype("string")
    return out


def prepare_dates_and_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["publication_date_raw"] = out["publication_date"]
    out["publication_date"] = pd.to_datetime(out["publication_date"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["year_from_publication_date"] = out["publication_date"].dt.year.astype("Int64")
    out["publication_date_invalid_flag"] = out["publication_date"].isna()
    out["year_mismatch_flag"] = (
        out["publication_date"].notna()
        & out["year"].notna()
        & out["year_from_publication_date"].notna()
        & out["year"].ne(out["year_from_publication_date"])
    )
    return out


def count_rows_with_year_before_2000(df: pd.DataFrame) -> int:
    analysis_year = df["year_from_publication_date"].where(
        df["year_from_publication_date"].notna(),
        df["year"],
    )
    return int((analysis_year.notna() & analysis_year.lt(2000)).sum())


def count_rows_with_dict_like_keywords(df: pd.DataFrame) -> int:
    return int(df["keywords"].apply(is_dict_like_keywords).sum())


def build_df_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    df_norm = pd.DataFrame(
        {
            "domain": out["domain"].astype("string").str.strip(),
            "category_norm": (
                out["category"]
                .astype("string")
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            ),
            "title": out["title"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip(),
            "description_norm": (
                out["description"]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            ),
            "publication_date_norm": out["publication_date"].dt.strftime("%Y-%m-%d").astype("string"),
            "keywords_norm": (
                out["keywords"]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            ),
            "year": out["year"],
        }
    )
    return df_norm.reset_index(drop=True)


def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=["link", "publication_date"], keep="first").copy()
    out = out.sort_values(["publication_date", "link"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    raw_df = load_input_data(INPUT_PATH)
    input_columns = list(raw_df.columns)

    input_rows = len(raw_df)
    rows_with_dict_like_keywords = count_rows_with_dict_like_keywords(normalized_df)
    normalized_df, normalize_domain = normalize_domain_column(normalized_df)
    normalized_df = normalize_domain
    normalized_df = normalize_keywords_column(normalized_df)
    normalized_df = add_description_word_count(normalized_df)
    prepared_df = prepare_dates_and_year(normalized_df)

    year_mismatch_rows = (
        prepared_df.loc[prepared_df["year_mismatch_flag"]]
        .sort_values(["publication_date", "link"], kind="mergesort")
        .reset_index(drop=True)
    )
    category_counts = build_unique_value_counts(prepared_df, "category")
    domain_counts = build_unique_value_counts(prepared_df, "domain")
    rows_with_year_before_2000 = count_rows_with_year_before_2000(prepared_df)

    cleaned_df = clean_basic(prepared_df)
    df_norm = build_df_norm(cleaned_df)

    print("Input rows:", input_rows)
    print("Rows with invalid publication_date:", int(prepared_df["publication_date_invalid_flag"].sum()))
    print("Rows with year < 2000:", rows_with_year_before_2000)
    print("Rows with dict-like/JSON-like keywords:", rows_with_dict_like_keywords)
    print("Rows in df_norm ready for equity split:", len(df_norm))
    print("Unique categories:", len(category_counts))
    print("Unique domains:", len(domain_counts))

 
    # print_preview("Category counts", category_counts, rows=len(category_counts))
    
if __name__ == "__main__":
    main()
