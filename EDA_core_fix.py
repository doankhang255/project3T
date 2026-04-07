from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

"""
EDA_core_fix.py

Main fixes compared with EDA_core.py
1. Normalize text more aggressively before regex matching.
2. Expand keyword coverage using real titles from sub_file.csv.
3. Use event_name as a secondary textual fallback when event_title is weak.
4. Prioritize positive/negative over neutral when a title matches both.
"""

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
# They are fallback signals when text-based matching is unresolved.
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

# Real titles in sub_file.csv contain many small typos and punctuation variants.
# Normalize them first so the regex map can stay readable.
TEXT_NORMALIZATION_RULES: Sequence[Tuple[str, str]] = (
    (r"\brregisters\b", "registers"),
    (r"\bregsiters\b", "registers"),
    (r"\bregisteres\b", "registers"),
    (r"\brepruchase\b", "repurchase"),
    (r"\badditonal\b", "additional"),
    (r"\bfiancial\b", "financial"),
    (r"\bdivdend\b", "dividend"),
    (r"\bextra ordinary\b", "extraordinary"),
    (r"\bpersonel\b", "personnel"),
    (r"\bpersonncel\b", "personnel"),
    (r"\bcompay\b", "company"),
    (r"\bdelistes\b", "delists"),
    (r"\boffcially\b", "officially"),
    (r"\boffficially\b", "officially"),
    (r"\boffically\b", "officially"),
    (r"\bofficiallys\b", "officially"),
    (r"\blabout\b", "labour"),
)

# Refined keyword map from real title phrases in sub_file.csv.
KEYWORD_MAP = {
    "positive": [
        r"\bcash dividend\b",
        r"\bdividend payment\b",
        r"\bpay(?:s)? cash dividend\b",
        r"\bbonus shares?\b",
        r"\bdividend shares?\b",
        r"\bdividend issue\b",
        r"\bbonus issue\b",
        r"\bshare buyback\b",
        r"\bbuy back\b",
        r"\bbuy treasury shares?\b",
        r"\brepurchase(?:s|d)?(?: treasury shares?)?\b",
        r"\bregisters? to repurchase\b",
        r"\balready repurchased\b",
        r"\bsuccessfully repurchased\b",
        r"\bbought\b.*\bshares\b",
        r"\bregisters? to buy\b",
        r"\bbecame (?:a )?major shareholders?\b",
        r"\braises? (?:its )?holding to\b",
        r"\bcontinued buying\b",
        r"\bremoved from (?:being |list of )?(?:alert|warning|control|supervision)\b",
        r"\beligible for margin trading\b",
        r"\bincreases? the foreign ownership limit\b",
        r"\braises? (?:the )?(?:rate of )?(?:limit )?foreign ownership\b",
        r"\bsuccessful repurchase\b",
    ],
    "negative": [
        r"\btrading suspension\b",
        r"\blisting suspension\b",
        r"\bforced delisting\b",
        r"\bdelist(?:s|ed|ing)?\b",
        r"\bcancel(?:s|led)? shares? registration\b",
        r"\bcancel(?:s|led)? (?:the )?registration\b",
        r"\bcancel(?:s|led)? shares?\b",
        r"\blast trading date\b",
        r"\bsuspending stock listing\b",
        r"\bunder (?:alert|warning|control|supervision)\b",
        r"\bineligible for margin trading\b",
        r"\bshare issuance\b",
        r"\bprivate placement\b",
        r"\bconvertible bond\b",
        r"\bright issue\b",
        r"\bissues? right for existing shareholders\b",
        r"\bloss(?:es)?\b",
        r"\bprofit decline\b",
        r"\bmiss(?:es|ed)? plan\b",
        r"\bpenalt(?:y|ies)\b",
        r"\bviolation\b",
        r"\bpostpone dividend\b",
        r"\bcancel dividend\b",
        r"\bsell treasury shares?\b",
        r"\bregisters? to sell treasury shares?\b",
        r"\bsold treasury shares?\b",
        r"\bdeliver(?:s|ed)? treasury shares?\b",
        r"\bempt(?:y|ied)\b.*\btreasury shares\b",
        r"\blists?\b.*\badditional shares\b",
        r"\bregisters?\b.*\badditional shares\b",
        r"\bissues? additional shares?\b",
        r"\badditional shares? for existing shareholders\b",
        r"\bissues? shares? for existing shareholders?\b",
        r"\bfailed to buy\b",
        r"\bfailed to sell\b",
        r"\bsell(?:s|ing| out)?\b.*\bshares\b",
        r"\bregisters? to sell\b",
        r"\bwas no longer as major shareholders?\b",
        r"\blowered (?:its )?holding to\b",
        r"\bdecreases? (?:the )?(?:rate of )?(?:foreign ownership|listed shares)\b",
        r"\bend(?:s)? the labour contract\b",
    ],
    "neutral": [
        r"\bquarterly results?\b",
        r"\bannual results?\b",
        r"\binterim results?\b",
        r"\bfinancial statements?\b",
        r"\baudited financial statements?\b",
        r"\bconsolidated financial statements?\b",
        r"\bboard resolution\b",
        r"\bbod\b.*\bresolution\b",
        r"\bboard of directors\b.*\bresolution\b",
        r"\bresolution dated\b",
        r"\bshareholders? meeting\b",
        r"\bgeneral meeting\b",
        r"\bannual general meeting\b",
        r"\bextraordinary shareholders? meeting\b",
        r"\bextraordinary general meeting\b",
        r"\bagm\b",
        r"\begm\b",
        r"\brecord date\b",
        r"\bballot\b",
        r"\bcircular ballot\b",
        r"\bex right date\b",
        r"\bname change\b",
        r"\bchange(?:s|d)?\b.*\bcompany\b.*\bname\b",
        r"\bchange(?:s|d)? (?:its |the company s )?name\b",
        r"\bchanges?\b.*\bhead office\b",
        r"\bchange(?:s|d)?\b.*\baddress\b",
        r"\bchange(?:s|d)?\b.*\b(?:seal|stamp)\b",
        r"\bofficially registers? shares? on\b",
        r"\bofficially registers? trading on\b",
        r"\bofficially registers? on\b",
        r"\bofficially lists? shares? on\b",
        r"\bofficially lists? on\b",
        r"\bofficially starts? trading\b",
        r"\bofficial trading on\b",
        r"\bstarts? trading on\b",
        r"\bshifts? listing (?:from|to)\b",
        r"\breturns? listing to\b",
        r"\blists? shares? on\b",
        r"\bregisters? shares? on\b",
        r"\bregisters? trading on\b",
        r"\bkey personnel changes?\b",
        r"\bkey personnel appointments?\b",
        r"\bcontrol member changes?\b",
        r"\bcontrol committee\b",
        r"\bappoints?\b",
        r"\bdismisses?\b",
        r"\bresigns?\b",
        r"\bleaves? from the position\b",
        r"\bretires?\b",
        r"\bchief accountant\b",
        r"\bdeputy ceo\b",
        r"\bboard member\b",
        r"\baudit contract\b",
        r"\bauditing firm\b",
        r"\bselects\b.*\bauditing\b",
        r"\bmerger and acquisition\b",
        r"\bforeign ownership limit is\b",
        r"\bthe rate of foreign ownership\b",
        r"\bsets up\b",
    ],
}

# event_name is much shorter and more controlled than title,
# so keep this map conservative and only include stable categories.
EVENT_NAME_KEYWORD_MAP = {
    "positive": [
        r"\bcash dividend\b",
        r"\bre listing\b",
    ],
    "negative": [
        r"\badditional listing\b",
        r"\blisting suspension\b",
    ],
    "neutral": [
        r"\bannual general meeting\b",
        r"\bextraordinary general meeting\b",
        r"\bagm resolution\b",
        r"\bquarterly results\b",
        r"\bfinal annual results\b",
        r"\byear interim result\b",
        r"\bbod meeting\b",
        r"\bkey personnel change\b",
        r"\bnew listing\b",
        r"\bexchange switching\b",
        r"\brecord date for ballot\b",
        r"\bname change\b",
        r"\bmerger and acquisition\b",
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
    """
    Lowercase, simplify punctuation, and fix common title typos.

    Why stronger normalization is useful here
    - sub_file.csv contains variants such as `BoD's`, `extra-ordinary`,
      `Offcially`, `repruchase`, `regsiters`.
    - Collapsing these variants first makes the regex map more stable.
    """
    if pd.isna(text):
        return ""

    normalized = str(text).lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9%]+", " ", normalized)

    for pattern, replacement in TEXT_NORMALIZATION_RULES:
        normalized = re.sub(pattern, replacement, normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_missing_title(text: object) -> bool:
    """
    Check whether `event_title` should be treated as missing.

    We consider both actual null values and empty/blank strings as missing.
    """
    if pd.isna(text):
        return True
    return normalize_text(text) == ""


def resolve_sentiment_matches(matched_labels: Iterable[str]) -> str:
    """
    Resolve matched labels into one final sentiment.

    Priority
    - If both positive and negative are present, keep `mixed`.
    - If a title is only neutral plus one polarity, keep the polarity.
    - If only neutral is present, return `neutral`.
    """
    labels = set(matched_labels)
    if not labels:
        return "unknown"

    polarity_labels = labels.intersection({"positive", "negative"})
    if len(polarity_labels) == 2:
        return "mixed"
    if len(polarity_labels) == 1:
        return next(iter(polarity_labels))
    if "neutral" in labels:
        return "neutral"
    return "unknown"


def match_sentiment_by_text(
    text: object,
    keyword_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Generic text-based sentiment matcher.

    Output
    - `positive`
    - `negative`
    - `neutral`
    - `mixed`
    - `unknown`
    - `None` if text is empty
    """
    keyword_map = keyword_map or KEYWORD_MAP
    normalized = normalize_text(text)

    if not normalized:
        return None

    matched_labels: List[str] = []
    for label, patterns in keyword_map.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            matched_labels.append(label)

    return resolve_sentiment_matches(matched_labels)


def match_sentiment_by_title(
    title: object,
    keyword_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Assign a sentiment label from the event title alone.

    This version is broader than EDA_core.py because it reflects the actual
    title variants observed in sub_file.csv.
    """
    keyword_map = keyword_map or KEYWORD_MAP
    return match_sentiment_by_text(title, keyword_map=keyword_map)


def match_sentiment_by_event_name(
    event_name: object,
    keyword_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """
    Conservative fallback matching using event_name.

    event_name is more templated than event_title, so only stable categories
    are included here.
    """
    keyword_map = keyword_map or EVENT_NAME_KEYWORD_MAP
    return match_sentiment_by_text(event_name, keyword_map=keyword_map)


def assign_sentiment_label(
    row: pd.Series,
    keyword_map: Optional[Dict[str, List[str]]] = None,
    fallback_to_event_code: bool = True,
) -> Optional[str]:
    """
    Final sentiment assignment for one row.

    Decision order in the fixed version
    1. Try title-based sentiment first.
    2. If title is unresolved, try a conservative event_name matcher.
    3. If still unresolved, optionally fall back to event_code prior.

    Why this differs from EDA_core.py
    - sub_file.csv shows many short procedural titles where the stable signal is
      still visible in `event_name`.
    - This reduces `unknown` labels without directly forcing every row to use
      event_code from the start.
    """
    title_label = match_sentiment_by_title(row.get("event_title"), keyword_map=keyword_map)

    if title_label in {"positive", "negative", "neutral", "mixed"}:
        return title_label

    event_name_label = match_sentiment_by_event_name(row.get("event_name"))
    if event_name_label in {"positive", "negative", "neutral"}:
        return event_name_label

    if fallback_to_event_code:
        event_code = row.get("event_code_norm") or row.get("event_code")
        event_code_label = EVENT_CODE_PRIOR.get(event_code, title_label)
        if event_code_label is not None:
            return event_code_label

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
    "EVENT_NAME_KEYWORD_MAP",
    "EVENT_CODE_PRIOR",
    "match_sentiment_by_title",
    "match_sentiment_by_event_name",
    "assign_sentiment_label",
    "run_eda_core",
    "keep_training_ready_labels",
    "build_quality_report",
    "build_sentiment_report",
]
