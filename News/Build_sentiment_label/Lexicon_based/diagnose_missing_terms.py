"""Mức 2 (mentor điểm 7 mở rộng): tìm nguyên nhân gốc của ~44% bài
Neutral_no_match, trong đó 35-45% thực ra là Positive/Negative theo ground
truth (xem diagnose_neutral_split.py).

2 câu hỏi cần trả lời bằng dữ liệu thật, không suy đoán:

  (A) Khớp từ có "quá chặt" không? Kiểm tra n-gram nào trong dictionary
      hiện tại KHÔNG BAO GIỜ khớp được lần nào trên toàn bộ 126.576 bài
      (candidate cho lỗi segmentation/n-gram cứng nhắc).

  (B) Từ nào đang THIẾU trong dictionary? Lấy đúng các bài trong ground
      truth gộp (ground_truth_combined.csv) bị chấm Neutral_no_match nhưng
      nhãn thật là Positive/Negative, tách token, đếm tần suất từ (unigram +
      bigram) CHƯA có trong dictionary - đây là ứng viên bổ sung ưu tiên cao
      nhất vì lấy trực tiếp từ bài bị chấm sai, không phải toàn corpus.

Output:
  data/dead_dictionary_terms.csv        (câu hỏi A)
  data/missing_term_candidates.csv      (câu hỏi B)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCORING_DIR.parents[2]
DATA_NEWS_DIR = PROJECT_ROOT / "data_news"
LEXICON_DIR = SCORING_DIR

DICTIONARY_PATH = LEXICON_DIR / "Scoring" / "data" / "weighted_dictionary.csv"
TOKENIZED_CORPUS_PATH = DATA_NEWS_DIR / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
ARTICLE_SCORES_PATH = LEXICON_DIR / "Scoring" / "data" / "article_scores.parquet"

GROUND_TRUTH_SETS = {
    "all_combined": PROJECT_ROOT / "data_news" / "ground_truth_combined.csv",
}

STOPWORDS_PATH = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "negation_cue_words.txt"
CLAUSE_BOUNDARY_PATH = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "clause_boundary_words.txt"

OUTPUT_DEAD_TERMS = LEXICON_DIR / "data" / "dead_dictionary_terms.csv"
OUTPUT_MISSING_CANDIDATES = LEXICON_DIR / "data" / "missing_term_candidates.csv"

# Từ chức năng cực phổ biến, không mang sentiment - loại khỏi candidate list
# để danh sách chỉ còn từ đáng xem xét (không phải danh sách đầy đủ, chỉ lọc
# thô những từ rõ ràng vô nghĩa nhất).
GENERIC_FUNCTION_WORDS = {
    "và", "của", "là", "có", "trong", "cho", "được", "này", "các", "với",
    "đã", "sẽ", "khi", "một", "những", "để", "đến", "từ", "theo", "về",
    "như", "vào", "tại", "trên", "ra", "cũng", "còn", "thì", "nên", "vì",
    "nếu", "mà", "nhưng", "hay", "hoặc", "đây", "đó", "nào", "gì", "ai",
    "bị", "phải", "không", "rất", "quá", "vẫn", "chỉ", "lại", "nữa",
    "công_ty", "doanh_nghiệp", "cổ_phiếu", "thị_trường", "năm", "ngày",
    "tháng", "quý", "vnd", "tỷ", "triệu", "đồng", "%",
}


def load_dictionary_term_set(dictionary_df: pd.DataFrame) -> set[str]:
    return set(dictionary_df["term"].tolist())


def load_word_set_from_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    cleaned = "\n".join(lines)
    items = [item.strip() for item in re.split(r"[,\n]", cleaned)]
    return {item for item in items if item}


def check_dead_terms(dictionary_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Câu hỏi (A): term nào trong dictionary KHÔNG BAO GIỜ match được lần
    nào trên toàn corpus (đếm bằng đúng logic n-gram greedy-longest, nhưng ở
    đây chỉ cần đếm xuất hiện thô của chuỗi n-gram, không cần ưu tiên dài
    nhất, vì mục đích là kiểm tra "chuỗi ký tự này có tồn tại trong corpus đã
    tokenize hay không" - nếu match thô = 0 thì chắc chắn match
    greedy-longest cũng = 0)."""
    by_ngram: dict[int, set[str]] = {}
    for row in dictionary_df.itertuples(index=False):
        by_ngram.setdefault(row.ngram_n, set()).add(row.term)

    match_counts: dict[str, int] = {term: 0 for term in dictionary_df["term"]}
    max_ngram = max(by_ngram.keys())

    for idx, row in enumerate(corpus_df.itertuples(index=False)):
        sentences = getattr(row, "Tokenize_content_sentences")
        for sent in sentences:
            tokens = list(sent)
            n_tokens = len(tokens)
            for n, term_set in by_ngram.items():
                if n > n_tokens:
                    continue
                for pos in range(n_tokens - n + 1):
                    candidate = " ".join(tokens[pos : pos + n])
                    if candidate in term_set:
                        match_counts[candidate] += 1
        if (idx + 1) % 20000 == 0:
            print(f"    ... đã quét {idx + 1}/{len(corpus_df)} bài (đếm match thô)")

    dead_df = dictionary_df.copy()
    dead_df["corpus_match_count"] = dead_df["term"].map(match_counts)
    dead_df = dead_df.sort_values("corpus_match_count")
    return dead_df


def mine_missing_candidates(
    corpus_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    dictionary_terms: set[str],
) -> pd.DataFrame:
    """Câu hỏi (B): trong các bài Neutral_no_match nhưng nhãn thật là
    Positive/Negative, đếm tần suất unigram + bigram CHƯA có trong
    dictionary, tách riêng theo nhãn thật (Positive vs Negative)."""
    counters = {"Positive": Counter(), "Negative": Counter()}
    n_articles = {"Positive": 0, "Negative": 0}

    for row in ground_truth_df.itertuples(index=False):
        label = row.sentiment
        if label not in counters:
            continue
        source_row_id = int(row.source_row_id)
        if source_row_id >= len(corpus_df):
            continue
        sentences = corpus_df.iloc[source_row_id]["Tokenize_content_sentences"]
        n_articles[label] += 1
        seen_in_article: set[str] = set()
        for sent in sentences:
            tokens = list(sent)
            for tok in tokens:
                seen_in_article.add(tok)
            for i in range(len(tokens) - 1):
                seen_in_article.add(f"{tokens[i]} {tokens[i+1]}")
        for term in seen_in_article:
            if term in dictionary_terms:
                continue
            if term in GENERIC_FUNCTION_WORDS:
                continue
            if term.isdigit():
                continue
            counters[label][term] += 1

    rows = []
    for label, counter in counters.items():
        total_articles = n_articles[label]
        for term, count in counter.most_common(150):
            rows.append(
                {
                    "true_label": label,
                    "term": term,
                    "ngram_n": term.count(" ") + 1,
                    "n_articles_with_term": count,
                    "n_missed_articles_this_label": total_articles,
                    "share_of_missed_articles": count / total_articles if total_articles else 0.0,
                }
            )

    return pd.DataFrame(rows).sort_values(["true_label", "n_articles_with_term"], ascending=[True, False])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc dictionary + corpus ...")
    dictionary_df = pd.read_csv(DICTIONARY_PATH, encoding="utf-8-sig")
    dictionary_terms = load_dictionary_term_set(dictionary_df)
    corpus_df = pd.read_parquet(TOKENIZED_CORPUS_PATH, columns=["Tokenize_content_sentences"])
    print(f"  dictionary: {len(dictionary_df)} term | corpus: {len(corpus_df)} bài")

    print("\n=== (A) Kiểm tra term 'chết' (0 match trên toàn corpus) ===")
    dead_df = check_dead_terms(dictionary_df, corpus_df)
    n_dead = int((dead_df["corpus_match_count"] == 0).sum())
    print(f"  {n_dead}/{len(dead_df)} term ({n_dead/len(dead_df)*100:.1f}%) KHÔNG BAO GIỜ match trên 126.576 bài.")
    print(dead_df.loc[dead_df["corpus_match_count"] == 0, ["term", "category", "ngram_n", "source"]].to_string(index=False))
    OUTPUT_DEAD_TERMS.parent.mkdir(parents=True, exist_ok=True)
    dead_df.to_csv(OUTPUT_DEAD_TERMS, index=False, encoding="utf-8-sig")
    print("  Đã lưu:", OUTPUT_DEAD_TERMS)

    print("\n=== (B) Đào ứng viên từ còn thiếu (từ bài Neutral_no_match nhưng nhãn thật P/N) ===")
    article_scores_df = pd.read_parquet(ARTICLE_SCORES_PATH, columns=["positive_score", "negative_score"]).reset_index(drop=True)
    article_scores_df["source_row_id"] = article_scores_df.index
    article_scores_df["no_match"] = (article_scores_df["positive_score"] == 0) & (article_scores_df["negative_score"] == 0)

    gt_frames = []
    for name, path in GROUND_TRUTH_SETS.items():
        df = pd.read_csv(path, encoding="utf-8-sig")[["source_row_id", "sentiment"]].copy()
        df["source_row_id"] = df["source_row_id"].astype(int)
        df["gt_set"] = name
        gt_frames.append(df)
    ground_truth_df = pd.concat(gt_frames, ignore_index=True)

    merged = ground_truth_df.merge(article_scores_df[["source_row_id", "no_match"]], on="source_row_id", how="left")
    missed_df = merged.loc[merged["no_match"] & merged["sentiment"].isin(["Positive", "Negative"])]
    print(f"  Số bài Neutral_no_match nhưng nhãn thật P/N: {len(missed_df)} / {len(merged)} bài ground truth")
    print(missed_df["sentiment"].value_counts().to_string())

    candidates_df = mine_missing_candidates(corpus_df, missed_df, dictionary_terms)
    n_unigram = int((candidates_df["ngram_n"] == 1).sum())
    n_bigram = int((candidates_df["ngram_n"] == 2).sum())
    print(f"\n  Trong top candidate: {n_unigram} unigram, {n_bigram} bigram (chưa lọc thủ công).")
    print("\n  Top 25 candidate cho Positive:")
    print(candidates_df.loc[candidates_df["true_label"] == "Positive"].head(25).to_string(index=False))
    print("\n  Top 25 candidate cho Negative:")
    print(candidates_df.loc[candidates_df["true_label"] == "Negative"].head(25).to_string(index=False))

    candidates_df.to_csv(OUTPUT_MISSING_CANDIDATES, index=False, encoding="utf-8-sig")
    print("\n  Đã lưu:", OUTPUT_MISSING_CANDIDATES)


if __name__ == "__main__":
    main()
