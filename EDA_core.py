from __future__ import annotations
import re
from typing import Dict, Iterable, List, Optional
import pandas as pd

# Expected columns based on the preview dataset.
REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "event_name",   
    "event_title",
    "event_code",
    "date_exr",
    "date_record",
    "date_issue",
    "exchange",
    "year",
}

TEXT_COLUMNS = ["symbol", "event_name", "event_title", "event_code", "exchange"]
DATE_COLUMNS = ["date", "date_exr", "date_record", "date_issue"]

# These maps are only weak priors.
# They are fallback signals when the title is missing.
# They do not replace the actual sentiment logic from the event title.
EVENT_CODE_PRIOR = {
    "DIV": "positive",
    "RETU": "positive",
    "MOVE": "neutral",
    "AGME": "neutral",
    "EGME": "neutral",
    "AGMR": "neutral",
    "BALLOT": "neutral",
    "KQQY": "neutral",
    "KQCT": "neutral",
    "KQSB": "neutral",
    "BCHA": "neutral",
    "BOME": "neutral",
    "DDALL": "neutral",
    "DDIND": "neutral",
    "DDINS": "neutral",
    "DDRP": "neutral",
    "AIS": "negative",
    "AMEN": "neutral",
    "ISS": "negative",
    "MA": "neutral",
    "NLIS": "neutral",
    "OTHE": None,
    "SUSP": "negative",
    "TS": "negative",
}

# Regex patterns are only a starter template.
# You should refine them after seeing more real titles.
KEYWORD_MAP = {
    "positive": [
        r"\bcash dividend\b",
        r"\bdividend payment\b",
        r"\bbonus share",
        r"\bshare buyback\b",
        r"\bbuy back\b",
        r"\bbuy treasury shares?\b",
        r"\bprofit exceeds plan\b",
        r"\bapproved listing\b",
        r"\bresume trading\b",
        r"\bremoved from warning\b",
        r"\bstrong growth\b",
        r"\bearnings beat\b",
    ],
    "negative": [
        r"\btrading suspension\b",
        r"\bdelisting\b",
        r"\bforced delisting\b",
        r"\bshare issuance\b",
        r"\bprivate placement\b",
        r"\bconvertible bond\b",
        r"\bloss\b",
        r"\bprofit decline\b",
        r"\bmiss(es|ed)? plan\b",
        r"\bpenalt(y|ies)\b",
        r"\bviolation\b",
        r"\bpostpone dividend\b",
        r"\bcancel dividend\b",
        r"\bsell treasury shares?\b",
    ],
    "neutral": [
        r"\bquarterly results?\b",
        r"\bannual results?\b",
        r"\bfinancial statements?\b",
        r"\bboard resolution\b",
        r"\bshareholder meeting\b",
        r"\bagm\b",
        r"\begm\b",
        r"\brecord date\b",
        r"\bex-right date\b",
        r"\bchange of name\b",
        r"\bchange of address\b",
        r"\binsider transaction\b",
        r"\bappoint(ment|ed)\b",
        r"\bresignation\b",
        r"\binformation disclosure\b",
    ],
}

def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the input DataFrame misses required columns."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dtypes into a predictable format for downstream EDA.

    Why this step matters
    - Date checks fail if columns are still strings.
    - Text rules fail if values are mixed objects / NaN / numbers.
    """
    out = df.copy()

    for col in TEXT_COLUMNS:
        out[col] = out[col].astype("string")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["date_exr", "date_record", "date_issue"]:
        out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def clean_implausible_issue_dates(
    df: pd.DataFrame,
    min_year: int = 2008,
    max_year: int = 2024,
) -> pd.DataFrame:
    """
    Drop rows with obviously wrong `date_issue` years.

    Note
    - Your notebook already noticed this issue.
    - Keep the thresholds configurable because the valid range depends on the
      real extraction window of your dataset.
    """
    out = df.copy()
    year_series = out["date_issue"].dt.year
    mask_bad = year_series.notna() & ((year_series < min_year) | (year_series > max_year))
    return out.loc[~mask_bad].copy()


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated rows based on the event identity.

    Why this key
    - `symbol` + `date` + `event_code` + `event_name` is a practical first pass.
    - If you later find duplicated titles under the same key, you can extend it.
    """
    return df.drop_duplicates(
        subset=["symbol", "date", "event_code", "event_name"],
        keep="first",
    ).copy()


def filter_invalid_exr_record_dates(df: pd.DataFrame, max_gap_days: int = 20) -> pd.DataFrame:
    """
    Remove rows where ex-right and record dates look inconsistent.

    Logic
    - If both dates exist, `date_exr` should not be later than `date_record`.
    - The gap should usually be under a practical threshold such as 20 days.

    Caution
    - This is a business rule, not a universal truth.
    - Keep it separate so you can relax it if domain validation says otherwise.
    """
    out = df.copy()

    invalid_order = (
        out["date_exr"].notna()
        & out["date_record"].notna()
        & (out["date_exr"] > out["date_record"])
    )

    invalid_gap = (
        out["date_exr"].notna()
        & out["date_record"].notna()
        & ((out["date_record"] - out["date_exr"]).dt.days >= max_gap_days)
    )

    return out.loc[~(invalid_order | invalid_gap)].copy()


def normalize_event_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a normalized event code column.

    Example
    - If `event_code == OTHE` but the title clearly mentions dividend,
      the row can be recoded to `DIV` for better downstream fallback rules.
    """
    out = df.copy()
    out["event_code_norm"] = out["event_code"].copy()

    dividend_mask = (
        out["event_code"].eq("OTHE")
        & out["event_title"].str.contains("dividend", case=False, na=False)
    )
    out.loc[dividend_mask, "event_code_norm"] = "DIV"

    return out


def normalize_text(text: object) -> str:
    """Lowercase and collapse whitespace for regex matching."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_missing_title(text: object) -> bool:
    """
    Check whether `event_title` should be treated as missing.

    We consider both actual null values and empty/blank strings as missing.
    """
    if pd.isna(text):
        return True
    return normalize_text(text) == ""


def match_sentiment_by_title(title: object, keyword_map: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """
    Assign a sentiment label from the event title alone.

    Output
    - `positive`
    - `negative`
    - `neutral`
    - `mixed`
    - `unknown`
    - `None` if title is empty

    Why keep `mixed` and `unknown`
    - `mixed` helps you inspect ambiguous titles manually.
    - `unknown` shows rule coverage is still incomplete.
    - For model training later, you can decide whether to keep or remove them.
    """
    keyword_map = keyword_map or KEYWORD_MAP
    text = normalize_text(title)

    if not text:
        return None

    matched_labels: List[str] = []
    for label, patterns in keyword_map.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matched_labels.append(label)

    matched_labels = sorted(set(matched_labels))
    if len(matched_labels) == 1:
        return matched_labels[0]
    if len(matched_labels) > 1:
        return "mixed"
    return "unknown"


def assign_sentiment_label(
    row: pd.Series,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    fallback_to_event_code: bool = True,
) -> Optional[str]:
    """
    Final sentiment assignment for one row.

    Decision order
    1. Try title-based sentiment first because the title is closer to meaning.
    2. Only if title is missing, optionally fall back to event code prior.

    This is one final label, not two separate label sets.

    Why this design
    - `event_title` is the primary source because it is closer to the actual
      meaning of the event.
    - `event_code` is only a backup signal for rows where the title is missing.
    - If the title exists but does not match the current rules, we keep
      `unknown` instead of forcing a fallback label from `event_code`.

    This also means you do not need to fill missing titles just to create
    synthetic text first. You can directly assign the final label with:
    title-first, event-code-fallback.
    """
    title = row.get("event_title")
    title_missing = is_missing_title(title)
    title_label = match_sentiment_by_title(title, keyword_map=keyword_map)

    if title_label in {"positive", "negative", "neutral", "mixed"}:
        return title_label

    if fallback_to_event_code and title_missing:
        event_code = row.get("event_code_norm") or row.get("event_code")
        return EVENT_CODE_PRIOR.get(event_code, title_label)

    return title_label


def add_sentiment_labels(
    df: pd.DataFrame,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    fallback_to_event_code: bool = True,
) -> pd.DataFrame:
    """
    Add one final `sentiment_label` to the dataset.

    Recommended usage for requirement 1
    - Keep `mixed` and missing labels for quality review.

    Recommended usage for requirement 2
    - Build a curated dataset that keeps only
      `positive`, `negative`, `neutral`.
    """
    out = df.copy()
    out["sentiment_label"] = out.apply(
        assign_sentiment_label,
        axis=1,
        keyword_map=keyword_map,
        fallback_to_event_code=fallback_to_event_code,
    )
    return out


def build_quality_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce compact tables for EDA review.

    These tables help explain the dataset in requirement 1.
    """
    missing_summary = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index(names="column")
        .sort_values("missing_count", ascending=False)
    )

    event_code_distribution = (
        df["event_code"]
        .value_counts(dropna=False)
        .rename_axis("event_code")
        .reset_index(name="count")
    )

    date_issue_year_distribution = (
        df["date_issue"]
        .dt.year
        .astype("Int64")
        .value_counts(dropna=False)
        .rename_axis("date_issue_year")
        .reset_index(name="count")
        .sort_values("date_issue_year", na_position="last")
    )

    return {
        "missing_summary": missing_summary,
        "event_code_distribution": event_code_distribution,
        "date_issue_year_distribution": date_issue_year_distribution,
    }


def build_sentiment_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Summaries focused on sentiment label coverage and balance."""
    label_distribution = (
        df["sentiment_label"]
        .value_counts(dropna=False)
        .rename_axis("sentiment_label")
        .reset_index(name="count")
    )

    by_event_code = (
        df.groupby(["event_code_norm", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["event_code_norm", "count"], ascending=[True, False])
    )

    by_symbol = (
        df.groupby(["symbol", "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["symbol", "count"], ascending=[True, False])
    )

    return {
        "label_distribution": label_distribution,
        "by_event_code": by_event_code,
        "by_symbol": by_symbol,
    }


def keep_training_ready_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the three target labels for the sentiment task.

    This subset is more suitable for requirement 2 model training.
    """
    valid = {"positive", "negative", "neutral"}
    return df.loc[df["sentiment_label"].isin(valid)].copy()


def run_eda_core(
    df_raw: pd.DataFrame,
    *,
    min_issue_year: int = 2008,
    max_issue_year: int = 2024,
    max_exr_record_gap_days: int = 20,
    fallback_to_event_code: bool = True,
) -> Dict[str, object]:
    """
    Full safe pipeline for requirement 1 and preparation for requirement 2.

    Returns
    - `data`: cleaned DataFrame with `event_code_norm` and `sentiment_label`
    - `training_data`: subset containing only positive/negative/neutral rows
    - `quality_report`: summary tables for EDA
    - `sentiment_report`: summary tables for sentiment inspection
    """
    validate_columns(df_raw)

    df = standardize_types(df_raw)
    df = clean_implausible_issue_dates(df, min_year=min_issue_year, max_year=max_issue_year)
    df = deduplicate_rows(df)
    df = filter_invalid_exr_record_dates(df, max_gap_days=max_exr_record_gap_days)
    df = normalize_event_code(df)
    df = add_sentiment_labels(df, fallback_to_event_code=fallback_to_event_code)

    quality_report = build_quality_report(df)
    sentiment_report = build_sentiment_report(df)
    training_data = keep_training_ready_labels(df)

    return {
        "data": df,
        "training_data": training_data,
        "quality_report": quality_report,
        "sentiment_report": sentiment_report,
    }


__all__ = [
    "KEYWORD_MAP",
    "EVENT_CODE_PRIOR",
    "run_eda_core",
    "keep_training_ready_labels",
    "build_quality_report",
    "build_sentiment_report",
]
