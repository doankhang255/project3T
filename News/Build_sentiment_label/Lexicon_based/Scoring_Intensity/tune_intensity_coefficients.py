"""Dò nhiều tổ hợp hệ số cho Cách 2 (intensity_weight) bằng quy trình
TUNE/HOLDOUT

Quy trình:
    1. Đọc split cố định từ Lexicon_based/data/ground_truth_tune_holdout_split.csv
       (730 bài Tune / 314 bài Holdout, split 1.044-dòng dùng chung toàn dự
       án - xem tune_negation_window.py).
    2. Dò 125 tổ hợp (5 base_config x 5 marker_delta x 5 intensifier_scale)
       CHỈ trên 730 bài Tune, chọn tổ hợp accuracy/macro-F1 cao nhất.
    3. Chấm lại 314 bài Holdout ĐÚNG 1 LẦN với tổ hợp đã chọn, so sánh với
       cấu hình hiện tại (A_current, marker_delta=1.0, intensifier_scale=1.0)
       cũng đo trên Holdout - đây là phép so sánh công bằng, khách quan.

3 nhóm hệ số được dò:
1. base_weight theo nguồn (final_seed..round4) - độ dốc của thang.
2. marker_delta - mức +/- khi từ chứa marker cực đoan/nhẹ.
3. intensifier_scale - hệ số nhân độ lệch của intensifier so với 1.0
   (VD hệ số gốc 1.3 cho "rất" -> lệch +0.3 so với 1.0; scale=2.0 nghĩa là
   lệch +0.6 -> hệ số thành 1.6).

Sau khi chọn được tổ hợp tốt nhất, BẮT BUỘC chạy lại toàn bộ 126,576 bài để
xem phân phối điểm có hợp lý không, không chỉ tin vào accuracy trên Tune.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCORING_DIR = Path(__file__).resolve().parent
SEED_SET_PREPARE_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare"
LATEST_SEED_ROUND_DIR = SEED_SET_PREPARE_DIR / "seed_round4"
PROVENANCE_PATH = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based" / "data" / "bootstrap" / "seed_provenance.csv"
NEGATION_PATH = SEED_SET_PREPARE_DIR / "negation_cue_words.txt"
CLAUSE_BOUNDARY_PATH = SEED_SET_PREPARE_DIR / "clause_boundary_words.txt"
TOKENIZED_CORPUS_PATH = PROJECT_ROOT / "data_news" / "data_tokenized" / "equity_news_tokenized_vncorenlp.parquet"
SPLIT_PATH = SCORING_DIR.parent / "data" / "ground_truth_tune_holdout_split.csv"
OUTPUT_SUMMARY_PATH = SCORING_DIR / "data" / "intensity_coefficient_tuning.csv"

# Cấu hình ĐANG DÙNG THẬT trong build_intensity_dictionary.py (BASE_WEIGHT_BY_SOURCE
# + MARKER_DELTA) - KHÁC với "A_current" trong BASE_WEIGHT_CONFIGS bên dưới
# (label đó đã lỗi thời, không còn khớp production kể từ lần chỉnh trước).
# Dùng đúng 3 giá trị này làm mốc so sánh công bằng trên Holdout.
PRODUCTION_BASE_WEIGHTS = {"final_seed": 5.0, "round1": 4.0, "round2": 3.0, "round3": 2.0, "round4": 1.0}
PRODUCTION_MARKER_DELTA = 0.0
PRODUCTION_INTENSIFIER_SCALE = 1.0

CATEGORY_NAMES = [
    "negative", "positive", "uncertainty", "litigious",
    "strong_modal", "weak_modal", "constraining",
]

EXTREME_MARKERS = {
    "tuyệt_đối", "hoàn_toàn", "tối_đa", "tối_thiểu", "tối_ưu", "chắc_chắn",
    "nghiêm_cấm", "nghiêm_ngặt", "nghiêm_trọng", "không_bao_giờ", "luôn_luôn",
    "luôn_đúng", "duy_nhất", "số_một", "cực_kỳ", "vô_cùng", "hết_sức",
    "thảm_họa", "khủng_hoảng", "phá_sản", "sụp_đổ", "vỡ_nợ", "bắt_buộc",
    "cấm", "tuyệt_vời", "hoàn_hảo", "kỷ_lục", "bùng_nổ", "khẳng_định",
    "dứt_khoát", "kiên_quyết", "quyết_liệt", "vượt_trội", "đột_phá",
    "cao_nhất", "thấp_nhất", "nhất_định", "hiển_nhiên", "đương_nhiên",
    "không_thể_tranh_cãi", "vô_song", "không_nghi_ngờ", "mạnh_mẽ_nhất",
    "tuyệt_nhiên_không", "tuyệt_đối_không", "hoàn_toàn_không",
}
MILD_MARKERS = {
    "hơi", "khá", "phần_nào", "tương_đối", "ít", "sơ_bộ", "tạm", "gần_như",
    "có_thể", "có_lẽ", "dường_như", "đôi_khi", "thỉnh_thoảng", "chưa_chắc",
    "không_hẳn", "hiếm_khi", "khả_dĩ", "biết_đâu", "chưa_hẳn", "có_phần",
    "nhẹ", "có_vẻ_như", "hình_như",
}
INTENSIFIER_BASE = {
    "hơi": 0.7, "khá": 0.7, "phần_nào": 0.7, "tương_đối": 0.7, "ít": 0.7,
    "rất": 1.3, "quá": 1.3, "lắm": 1.3, "thật": 1.3, "thật_sự": 1.3,
    "cực_kỳ": 1.6, "vô_cùng": 1.6, "hết_sức": 1.6, "tuyệt_đối": 1.6,
    "hoàn_toàn": 1.6, "đặc_biệt": 1.6,
    "đáng_kể": 1.4, "nghiêm_trọng": 1.4, "mạnh_mẽ": 1.4, "sâu_sắc": 1.4,
    "rõ_rệt": 1.4,
}

NEGATION_WINDOW = 4

BASE_WEIGHT_CONFIGS: dict[str, dict[str, float]] = {
    "A_moderate": {"final_seed": 4.0, "round1": 3.5, "round2": 3.0, "round3": 2.5, "round4": 2.0},
    "B_production": {"final_seed": 5.0, "round1": 4.0, "round2": 3.0, "round3": 2.0, "round4": 1.0},  # = PRODUCTION_BASE_WEIGHTS
    "C_flat":     {"final_seed": 3.0, "round1": 2.8, "round2": 2.6, "round3": 2.4, "round4": 2.2},
    "D_steepest": {"final_seed": 4.0, "round1": 3.0, "round2": 2.0, "round3": 1.0, "round4": 0.5},
    "E_uniform":  {"final_seed": 3.0, "round1": 3.0, "round2": 3.0, "round3": 3.0, "round4": 3.0},
}
MARKER_DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INTENSIFIER_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


def load_seed_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    items = [item.strip() for item in re.split(r"[,\n]", text)]
    return [item for item in items if item]


def load_word_set_from_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    cleaned = "\n".join(lines)
    items = [item.strip() for item in re.split(r"[,\n]", cleaned)]
    return {item for item in items if item}


def load_source_lookup() -> dict[tuple[str, str], str]:
    provenance_df = pd.read_csv(PROVENANCE_PATH, encoding="utf-8-sig", dtype=str)
    provenance_df["round"] = provenance_df["round"].astype(int)
    provenance_df = provenance_df.sort_values("round")
    lookup: dict[tuple[str, str], str] = {}
    for row in provenance_df.itertuples(index=False):
        key = (row.category, row.term)
        if key not in lookup:
            lookup[key] = f"round{row.round}"
    return lookup


def is_negated(tokens, pos, negation_words, clause_boundary_words, window):
    window_start = max(0, pos - window)
    for i in range(pos - 1, window_start - 1, -1):
        tok = tokens[i]
        if tok in negation_words:
            return True
        if tok in clause_boundary_words:
            return False
    return False


def score_ground_truth_articles(
    gt_sentences: dict[int, list],
    gt_total_tokens: dict[int, int],
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]],
    max_ngram: int,
    negation_words: set[str],
    clause_boundary_words: set[str],
    intensifier_multipliers: dict[str, float],
) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for row_id, sentences in gt_sentences.items():
        category_sums = {c: 0.0 for c in CATEGORY_NAMES}
        for sent in sentences:
            tokens = list(sent)
            n_tokens = len(tokens)
            pos = 0
            while pos < n_tokens:
                matched_len = 0
                for n in range(min(max_ngram, n_tokens - pos), 0, -1):
                    term_map = by_ngram.get(n)
                    if not term_map:
                        continue
                    candidate = " ".join(tokens[pos : pos + n])
                    hits = term_map.get(candidate)
                    if not hits:
                        continue
                    negated = is_negated(tokens, pos, negation_words, clause_boundary_words, NEGATION_WINDOW)
                    sign = -1.0 if negated else 1.0
                    multiplier = intensifier_multipliers.get(tokens[pos - 1], 1.0) if pos > 0 else 1.0
                    for category, weight in hits:
                        category_sums[category] += weight * multiplier * sign
                    matched_len = n
                    break
                pos += matched_len if matched_len else 1
        total_tokens = gt_total_tokens[row_id] or 1
        results[row_id] = {c: category_sums[c] / total_tokens for c in CATEGORY_NAMES}
    return results


def evaluate(scores: dict[int, dict[str, float]], gt_df: pd.DataFrame) -> tuple[float, float]:
    y_true = []
    y_pred = []
    for _, row in gt_df.iterrows():
        row_id = int(row["source_row_id"])
        if row_id not in scores:
            continue
        pos_score = scores[row_id]["positive"]
        neg_score = scores[row_id]["negative"]
        if pos_score > neg_score:
            pred = "Positive"
        elif pos_score < neg_score:
            pred = "Negative"
        else:
            pred = "Neutral"
        y_true.append(row["sentiment"])
        y_pred.append(pred)

    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    accuracy = (y_true_s == y_pred_s).mean()

    f1_scores = []
    for label in ["Positive", "Neutral", "Negative"]:
        tp = ((y_true_s == label) & (y_pred_s == label)).sum()
        fp = ((y_true_s != label) & (y_pred_s == label)).sum()
        fn = ((y_true_s == label) & (y_pred_s != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return accuracy, macro_f1


def build_weight_table(
    base_weights: dict[str, float],
    marker_delta: float,
    term_rows: list[tuple[str, str, int]],
    source_lookup: dict[tuple[str, str], str],
) -> tuple[dict[int, dict[str, list[tuple[str, float]]]], int]:
    by_ngram: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for category, term, ngram_n in term_rows:
        source = source_lookup.get((category, term), "final_seed")
        base = base_weights[source]
        subtokens = term.split(" ")
        adjustment = 0.0
        for tok in subtokens:
            if tok in EXTREME_MARKERS:
                adjustment += marker_delta
            elif tok in MILD_MARKERS:
                adjustment -= marker_delta
        weight = max(1.0, min(5.0, base + adjustment))
        by_ngram.setdefault(ngram_n, {}).setdefault(term, []).append((category, weight))
    return by_ngram, max(by_ngram.keys())


def score_and_eval_combo(
    base_weights: dict[str, float],
    marker_delta: float,
    intensifier_scale: float,
    term_rows: list[tuple[str, str, int]],
    source_lookup: dict[tuple[str, str], str],
    gt_sentences: dict[int, list],
    gt_total_tokens: dict[int, int],
    gt_df: pd.DataFrame,
    negation_words: set[str],
    clause_boundary_words: set[str],
) -> tuple[float, float]:
    by_ngram, max_ngram = build_weight_table(base_weights, marker_delta, term_rows, source_lookup)
    intensifier_multipliers = {tok: 1.0 + (mult - 1.0) * intensifier_scale for tok, mult in INTENSIFIER_BASE.items()}
    scores = score_ground_truth_articles(
        gt_sentences, gt_total_tokens, by_ngram, max_ngram, negation_words, clause_boundary_words, intensifier_multipliers
    )
    return evaluate(scores, gt_df)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc term list (seed_round4) + provenance (nguồn gốc) ...")
    source_lookup = load_source_lookup()
    term_rows: list[tuple[str, str, int]] = []  # (category, term, ngram_n)
    for category in CATEGORY_NAMES:
        for term in load_seed_words(LATEST_SEED_ROUND_DIR / f"{category}_word.txt"):
            term_rows.append((category, term, term.count(" ") + 1))
    print(f"  {len(term_rows)} term")

    negation_words = load_word_set_from_file(NEGATION_PATH)
    clause_boundary_words = load_word_set_from_file(CLAUSE_BOUNDARY_PATH)

    print("Đọc split Tune/Holdout ...")
    split_df = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig")
    tune_gt_df = split_df.loc[split_df["split"] == "tune", ["source_row_id", "sentiment"]].reset_index(drop=True)
    holdout_gt_df = split_df.loc[split_df["split"] == "holdout", ["source_row_id", "sentiment"]].reset_index(drop=True)
    print(f"  Tune: {len(tune_gt_df)} bài | Holdout: {len(holdout_gt_df)} bài")

    print("Đọc corpus, trích riêng các bài trong Tune + Holdout ...")
    corpus_df = pd.read_parquet(TOKENIZED_CORPUS_PATH)
    all_row_ids = pd.concat([tune_gt_df["source_row_id"], holdout_gt_df["source_row_id"]]).astype(int).tolist()
    sentences_by_id = {rid: corpus_df.iloc[rid]["Tokenize_content_sentences"] for rid in all_row_ids}
    total_tokens_by_id = {rid: corpus_df.iloc[rid]["total_tokenizer"] for rid in all_row_ids}
    del corpus_df
    tune_sentences = {rid: sentences_by_id[rid] for rid in tune_gt_df["source_row_id"].astype(int)}
    tune_total_tokens = {rid: total_tokens_by_id[rid] for rid in tune_gt_df["source_row_id"].astype(int)}
    holdout_sentences = {rid: sentences_by_id[rid] for rid in holdout_gt_df["source_row_id"].astype(int)}
    holdout_total_tokens = {rid: total_tokens_by_id[rid] for rid in holdout_gt_df["source_row_id"].astype(int)}

    results = []
    total_combos = len(BASE_WEIGHT_CONFIGS) * len(MARKER_DELTAS) * len(INTENSIFIER_SCALES)
    print(f"\n=== BƯỚC 1: dò {total_combos} tổ hợp TRÊN TUNE ({len(tune_gt_df)} bài) ===")
    done = 0
    for base_name, base_weights in BASE_WEIGHT_CONFIGS.items():
        for marker_delta in MARKER_DELTAS:
            for scale in INTENSIFIER_SCALES:
                accuracy, macro_f1 = score_and_eval_combo(
                    base_weights, marker_delta, scale, term_rows, source_lookup,
                    tune_sentences, tune_total_tokens, tune_gt_df, negation_words, clause_boundary_words,
                )
                results.append(
                    {
                        "base_config": base_name,
                        "marker_delta": marker_delta,
                        "intensifier_scale": scale,
                        "accuracy_tune": accuracy,
                        "macro_f1_tune": macro_f1,
                    }
                )
                done += 1
                if done % 20 == 0:
                    print(f"  ... đã dò {done}/{total_combos} tổ hợp")

    results_df = pd.DataFrame(results).sort_values(["accuracy_tune", "macro_f1_tune"], ascending=False)
    print("\n=== TOP 10 TỔ HỢP TỐT NHẤT TRÊN TUNE (theo accuracy, rồi macro_f1) ===")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(
        f"\nTỐT NHẤT TRÊN TUNE: base_config={best['base_config']}, marker_delta={best['marker_delta']}, "
        f"intensifier_scale={best['intensifier_scale']} "
        f"-> accuracy_tune={best['accuracy_tune']:.4f}, macro_f1_tune={best['macro_f1_tune']:.4f}"
    )

    print(f"\n=== BƯỚC 2: chấm Holdout ({len(holdout_gt_df)} bài) ĐÚNG 1 LẦN, so sánh tổ hợp tốt nhất vs cấu hình production hiện tại ===")
    acc_best_holdout, f1_best_holdout = score_and_eval_combo(
        BASE_WEIGHT_CONFIGS[best["base_config"]], best["marker_delta"], best["intensifier_scale"],
        term_rows, source_lookup, holdout_sentences, holdout_total_tokens, holdout_gt_df,
        negation_words, clause_boundary_words,
    )
    acc_prod_holdout, f1_prod_holdout = score_and_eval_combo(
        PRODUCTION_BASE_WEIGHTS, PRODUCTION_MARKER_DELTA, PRODUCTION_INTENSIFIER_SCALE,
        term_rows, source_lookup, holdout_sentences, holdout_total_tokens, holdout_gt_df,
        negation_words, clause_boundary_words,
    )
    print(
        f"  Tổ hợp chọn từ Tune ({best['base_config']}/{best['marker_delta']}/{best['intensifier_scale']}) "
        f"trên Holdout: accuracy={acc_best_holdout:.4f}  macro_f1={f1_best_holdout:.4f}"
    )
    print(
        f"  Production hiện tại (base=round-giảm-dần 5-4-3-2-1, marker_delta=0.0, intensifier_scale=1.0) "
        f"trên Holdout: accuracy={acc_prod_holdout:.4f}  macro_f1={f1_prod_holdout:.4f}"
    )

    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\nĐã lưu:", OUTPUT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
