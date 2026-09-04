# Scoring_Intensity (Cách 2 — trọng số nội tại, không dùng PMI)

Song song với `../Scoring/` (Cách 1: `weight = score_centered` từ PMI, đo
mức độ đồng-xuất-hiện với seed TRONG corpus). Folder này là **Cách 2**:
`weight = intensity_weight` — trọng số NỘI TẠI của từ (giống ý tưởng VADER
có sẵn điểm cho từng từ, KHÔNG phụ thuộc corpus).

## Công thức

```
S_cat = ( Σᵢ intensity_weight(wᵢ) × intensifier_multiplier(wᵢ) × sign(wᵢ) ) / N_total
```

- `intensity_weight`: xem `build_intensity_dictionary.py` — 2 lớp:
  1. **base_weight theo nguồn** (final_seed=4.0 → round4=2.0, "từ càng gốc/càng
     trung tâm thì trọng số càng cao", đúng ý tưởng ban đầu: "tăng_trưởng"
     final_seed sẽ cao hơn 1 từ tìm được ở round4).
  2. **marker_adjustment**: +1.0 nếu từ chứa 1 marker cực đoan/tuyệt đối
     (VD "tuyệt_đối", "phá_sản", "cực_kỳ"), -1.0 nếu chứa marker nhẹ/dè dặt
     (VD "hơi", "tương_đối"). Kết quả cuối chặn trong [1.0, 5.0].
- `intensifier_multiplier`: nếu token NGAY TRƯỚC vị trí match là 1 từ trong
  `Intensifier_words/intensifier_words.txt`, nhân thêm hệ số (0.7/1.3/1.4/1.6
  theo nhóm) — xem `score_articles_intensity.py`.
- `sign`: giống hệt Cách 1 — phủ định (`negation_cue_words.txt`) + chặn ranh
  giới mệnh đề (`clause_boundary_words.txt`), cửa sổ 4 token.

## File (2 luồng)

**Luồng 1 - xây dựng**: `build_intensity_dictionary.py` → `data/intensity_dictionary.csv`,
rồi `score_articles_intensity.py` → `data/article_scores_intensity.parquet`.

**Luồng 2 - kiểm tra độ chính xác**: `classify_and_evaluate_intensity.py` →
đối chiếu `ground_truth_labeled.csv`, in accuracy/confusion matrix/F1, lưu
`data/evaluation_vs_ground_truth_intensity.csv` — **so trực tiếp được** với
kết quả Cách 1 (PMI) ở `../Scoring/data/evaluation_vs_ground_truth.csv` vì
dùng chung `ground_truth_labeled.csv`, chung công thức khung (chỉ khác
nguồn trọng số).

## GIỚI HẠN QUAN TRỌNG (đọc trước khi tin dùng)

1. **`base_weight` theo round và `marker_adjustment` (+1.0/-1.0) là số TỰ
   CHỌN**, không phải do người/LLM chấm riêng từng từ trong 1729 từ (không
   khả thi thủ công ở quy mô này) - đây là 1 QUY TẮC (rule-based) mô phỏng
   lại kiểu phán đoán đã dùng suốt phiên review, chưa phải "1729 từ đã được
   chấm điểm độc lập từng từ".
2. **Hệ số nhân intensifier (0.7/1.3/1.4/1.6)** cũng là lựa chọn ban đầu,
   chưa hiệu chỉnh thực nghiệm (khác VADER có hệ số đo trên người đánh giá
   thật).
3. Đúng như đã giải thích: **thang điểm tuyệt đối (1-5) không ảnh hưởng
   accuracy** vì Cách 3 gán nhãn chỉ so sánh positive_score với negative_score
   (scale đều không đổi thứ tự so sánh) - cái quyết định accuracy là **tỷ lệ
   tương đối giữa các từ**, cần kiểm chứng bằng `classify_and_evaluate_intensity.py`
   so với Cách 1, không có "đúng sẵn" để tra cứu.
