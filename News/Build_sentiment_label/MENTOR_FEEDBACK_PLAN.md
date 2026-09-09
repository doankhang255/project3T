# Kế hoạch hành động sau feedback của mentor (Mike Nguyen)

Ngày ghi nhận feedback: 2026-09-05 (nhắn qua Slack). Tài liệu này áp lại 7 điểm
feedback thành việc cụ thể, theo từng nhánh (`Lexicon_based/`, `Traditional_ML/`,
`Transfer_Learning/`), để không làm tràn lan cùng lúc.

**Kết luận chung của mentor**: "Nhìn chung hướng làm đúng và tốt. Cứ tập trung
vào ground truth và kiểm định ở tầng chỉ số ngày, đừng tối ưu thêm hệ số trên
152 bài nữa."

---

## A. Sửa lỗi thống kê ở kết quả hiện có (điểm 1–2) — `Lexicon_based/`

**Vấn đề mentor chỉ ra**: với n=152, sai số chuẩn của accuracy ~4 điểm %, nên
61,2% / 63,2% / 63,8% (các bước trong `Scoring/` và `Scoring_Intensity/`) về
mặt thống kê là **như nhau** — không nên kết luận "longest-match kém hơn"
hay "Cách 2 tốt hơn Cách 1". Nặng hơn: 125 tổ hợp hệ số (Cách 2),
`NEGATION_WINDOW`, neutral margin — đều được **chọn trên chính 152 bài dùng
để báo cáo kết quả** → 63,2% là con số in-sample, không phải ước lượng
khách quan.

- [ ] Chia 152 bài ground truth thành tune/holdout (hoặc nested k-fold CV) —
      áp lại cho: `negation_window` (`compare_negation_windows.py`), neutral
      margin (`tune_neutral_margin.py`, hiện không dùng), 125 tổ hợp hệ số
      Cách 2 (`tune_intensity_coefficients.py`).
- [ ] Báo lại accuracy trên phần holdout/CV thay vì con số tune-trên-cùng-tập.
- [ ] Thêm caveat sai số chuẩn (~4pp với n=152) vào `METHODOLOGY_SUMMARY.qmd`
      mỗi chỗ so sánh % giữa các bước.

**Trạng thái**: chưa bắt đầu — làm được ngay, không cần dữ liệu mới.

---

## B. Mở rộng ground truth (điểm 3) — cần phối hợp người khác

**Mục tiêu**: 800–1000 bài (hiện có 152).

- [ ] Lấy mẫu **có chiến lược** thay vì ngẫu nhiên: phân tầng theo thời gian
      + theo ngành.
- [ ] Thêm active learning: ưu tiên gán các bài model đang "phân vân" nhất
      (VD `|positive_score − negative_score|` nhỏ, hoặc không match từ nào).
- [ ] Tuyển ≥2 người gán độc lập trên phần chồng lấn (mentor khuyến khích 4
      người), tính **Cohen's kappa** — biết trần (ceiling) bài toán là bao
      nhiêu. Nếu người gán chỉ đồng thuận ~70%, coi như accuracy 63% hiện tại
      đã gần trần.

**Trạng thái**: cần tự sắp xếp với Hải/Mến/lab mate. Có thể quay lại nhờ hỗ
trợ code (script lấy mẫu phân tầng, script tính kappa) khi đến bước đó.

---

## C. Kiểm định cấp-ngày kiểu Tetlock (điểm 4) — ĐÃ XONG bản đầu, xem kết luận

**Ý tưởng**: không cần chờ ground truth lớn — sai số cấp-bài triệt tiêu bớt
khi tổng hợp theo ngày trên toàn bộ 126.576 bài, rồi hồi quy với lợi suất
VN-Index (thiết kế Tetlock 2007, có kiểm soát lợi suất trễ + khối lượng giao
dịch). Nếu chỉ số ngày có sức giải thích, phương pháp coi như hoạt động dù
accuracy cấp-bài mới 63%. Đây cũng là cách so Cách 1 vs Cách 2 khách quan
hơn (cỡ mẫu lớn hơn nhiều so với 152 bài).

**Cấu trúc thư mục hiện tại** (đọc lại `REF/Tetlock_Media_Sentiment_JF.pdf`
để đối chiếu thiết kế trước khi làm — xem docstring từng file):
- `News/Build_sentiment_index/` — xây chỉ số sentiment theo kỳ (ngày/tuần),
  đọc thẳng `Lexicon_based/Scoring*/data/article_scores*.parquet` (không cần
  `predicted_label`, chỉ cần `net_sentiment_score`):
  - `build_sentiment_index_daily.py` / `build_sentiment_index_weekly.py` — trung bình đơn giản positive-negative (Cách 1 PMI, Cách 2 intensity).
  - `build_sentiment_index_pca.py` — **chỉ số PCA kiểu Tetlock thật** trên 7 category LM (xem mục "PCA factor" bên dưới).
- `News_Vnindex/Common/` — hàm merge dùng chung cho mọi method (`merge_vnindex_daily_with_sentiment.py` có `map_to_next_trading_date` map tin cuối tuần/lễ vào phiên kế tiếp; `merge_vnindex_weekly_with_sentiment.py` tương ứng cấp tuần).
- `News_Vnindex/Lexicon/daily/` — pipeline cấp ngày (tự chứa, không import từ `weekly/`): `build_vnindex_daily_abnormal_return.py`, `vnindex_daily_predictive_regression.py` (multi-lag + dummy `near_tet` + Newey-West), cộng các script chẩn đoán (`check_predictor_multicollinearity.py`, `visualize_coefficient_covariance.py`, `plot_sentiment_lag_coefficients.py`, `plot_ols_fit_vs_actual.py`, `verify_abnormal_return.py`, `diagnose_model_r_squared.py`, `vnindex_daily_predictive_regression_covid_control.py`).
- `News_Vnindex/Lexicon/weekly/` — pipeline cấp tuần (bản đơn giản 1-lag có sẵn từ trước, đã sửa path cho khớp Lexicon_based mới): `build_vnindex_weekly_return.py`, `build_vnindex_weekly_abnormal_return.py`, `vnindex_weekly_predictive_regression.py`.
- 4 method test song song ở cả 2 tần suất: `pmi`, `intensity`, `pca_pmi`, `pca_intensity`.

### Kết quả DAILY (multi-lag, Newey-West, đã dọn VIF/near_tet/volatility-lag)

**Không tìm thấy bằng chứng vững chắc** ở cả 4 method. Lag1 (tức thời) không
có ý nghĩa (p 0,25–0,64). Kiểm định tổng lag2-5 (đảo chiều) không có ý nghĩa
(p 0,21–0,86). Lag4 có ý nghĩa riêng lẻ (p<0,02) ở PMI/Intensity nhưng **biến
mất hoàn toàn với PCA_PMI/PCA_Intensity** (p 0,75–0,98) — bằng chứng khá rõ
lag4 là nhiễu multiple-testing, không phải tín hiệu thật (nếu thật, phải
xuất hiện lại với chỉ số sentiment khác). Robustness check thêm dummy
COVID-19 (23/1–30/4/2020) không đổi kết luận. R² toàn mô hình chỉ 1,36% —
đã tách theo nhóm biến (`diagnose_model_r_squared.py`): phần lớn (73%) đến
từ dummy mùa vụ/volatility, **không phải từ sentiment lẫn lịch sử return** —
cho thấy return cấp ngày VN-Index nói chung rất khó dự đoán từ bất kỳ nguồn
nào ở đây, không riêng sentiment.

### Kết quả WEEKLY — phát hiện đáng chú ý nhất

Ở horizon **4 tuần**, `future_ret_4w` (return thô, chưa điều chỉnh abnormal)
cho **p ≈ 0,05–0,06 nhất quán ở cả 4 method độc lập** (PMI 0,057, Intensity
0,051, PCA_PMI 0,052, PCA_Intensity 0,054) — mức nhất quán chưa từng thấy ở
các kiểm định khác trong toàn bộ quá trình (so với lag4 ở daily chỉ xuất
hiện 2/4 method). R² cũng cao hơn hẳn daily (~2% so với 1,36%). Tuy nhiên 2
biến thể abnormal-return ở 4 tuần (`future_abnormal_rolling_ret_4w`,
`future_abnormal_ar1_ret_4w`) chỉ có ý nghĩa borderline ở PMI/Intensity
(p 0,057–0,081) và **mất ý nghĩa với PCA** (p 0,18–0,28) — nên "future_ret_4w
thô" nhất quán nhưng chưa rõ có do sentiment thật hay do 1 phần chưa lọc hết
(bản weekly hiện tại vẫn là bản đơn giản 1-lag, chưa nâng cấp multi-lag +
dummy mùa vụ như bản daily).

### PCA factor (kiểu Tetlock thật, thay trung bình đơn giản)

`build_sentiment_index_pca.py` — PCA trên 7 category LM (thay vì chỉ
positive-negative). **Phát hiện quan trọng**: component 1 có `positive_score`
VÀ `negative_score` cùng hệ số DƯƠNG (không đối lập như Tetlock, nơi Positive
load âm, Negative load dương) — component 1 ở đây giống chỉ số "mật độ ngôn
từ tài chính nói chung" (weak_modal/strong_modal chi phối mạnh nhất) hơn là
trục "bi quan-lạc quan" sạch. Giải thích được 25-43% phương sai 7 category
(tùy ngày/tuần). Vẫn đưa vào test thực nghiệm — kết quả xem 2 mục trên.

### Weekly multi-lag (nâng cấp, `vnindex_weekly_predictive_regression_multilag.py`)

Nâng cấp weekly lên đúng đặc tả daily (5-lag sentiment/target + 1 lag volume
+ dummy `near_tet` + `volatility_12w` đã sửa `.shift(1)` tránh look-ahead,
Newey-West, kiểm định tổng lag2-5) — import lại `newey_west_covariance` từ
bản đơn giản CÙNG thư mục `weekly/` (không xuyên `daily/`↔`weekly/`).

**Kết quả: tín hiệu "future_ret_4w nhất quán 4 method" từ bản đơn giản
KHÔNG được xác nhận lại.** PMI/Intensity: không còn gì (mọi p 0,18–0,87).
PCA_PMI/PCA_Intensity: lag2 âm + lag3 dương có ý nghĩa/borderline riêng lẻ
(p 0,02–0,07) nhưng gần như **triệt tiêu lẫn nhau** khi cộng lại → kiểm
định tổng lag2-5 không có ý nghĩa (p 0,68–0,85). Pattern chỉ xuất hiện với
PCA, không với PMI/Intensity — cùng dấu hiệu kém tin cậy đã thấy với lag4 ở
daily (không nhất quán qua các cách tính sentiment khác nhau).

### Kết luận tổng thể cho mentor (sau khi thử daily rigorous + weekly simple
### + weekly multi-lag + PCA factor — 4 hướng kiểm định độc lập)

**Chưa tìm được bằng chứng vững chắc, nhất quán về liên hệ sentiment ↔ lợi
suất VN-Index** ở bất kỳ tần suất/cách tính nào khi kiểm tra đủ nghiêm ngặt
(multi-lag + dummy mùa vụ + Newey-West + nhiều cách tính sentiện độc lập).
Mọi tín hiệu "có vẻ có ý nghĩa" phát hiện dọc đường (lag4 daily, future_ret_4w
weekly đơn giản, lag2/3 weekly PCA) đều **biến mất hoặc không nhất quán**
khi kiểm tra chéo bằng cách tính sentiment khác — đúng dấu hiệu nhiễu do
kiểm định nhiều lần (đã chạy tổng cộng hàng trăm phép kiểm định qua cả quá
trình), không phải tín hiệu kinh tế thật ổn định.

**Điều này KHÔNG có nghĩa phương pháp Lexicon-Based thất bại** — chỉ có
nghĩa dữ liệu/kiểm định hiện tại (ground truth 152 bài, corpus tin thưa
2010-2013, sentiment cấp-vĩ-mô có thể bị pha loãng bởi tin công ty đơn lẻ)
chưa đủ mạnh để phát hiện tín hiệu, nếu có. Việc cần làm tiếp theo quan
trọng nhất vẫn là **mục B (ground truth 800-1000 bài)** — không nên đầu tư
thêm vào việc dò thêm biến thể hồi quy nữa cho tới khi có ground truth lớn
hơn để biết chính accuracy cấp-bài đang ở đâu so với trần con người.

- [x] Viết script mới: tổng hợp `article_scores_labeled.parquet` (Cách 1) và
      `article_scores_intensity_labeled.parquet` (Cách 2) theo ngày → sentiment
      index hàng ngày kiểu mới, KHÔNG đụng vào `market_sentiment_index_daily.parquet`
      cũ (folder mới `Lexicon_based/Market_Validation/`, self-contained).
- [x] Merge với `vnindex_eda_output.csv` theo ngày (3.468/5.037 ngày khớp
      được giao dịch thật).
- [x] Hồi quy lợi suất VN-Index ~ sentiment ngày (Newey-West HAC, import
      thẳng hàm từ `News_Vnindex/vnindex_weekly_predictive_regression.py`,
      không viết lại) — đã đọc lại `REF/Tetlock_Media_Sentiment_JF.pdf` để
      đối chiếu thiết kế, xem `Market_Validation/README.md` mục "Khác biệt".
- [x] Dùng kết quả này so sánh Cách 1 vs Cách 2 — **kết quả lần chạy đầu
      (bản đơn giản, 1 lag) chưa có ý nghĩa thống kê ở cả 4 tổ hợp** (p >
      0,18) — chưa kết luận được gì, cần nâng cấp lên multi-lag + dummy
      mùa vụ (đúng bản gốc Tetlock) trước khi kết luận không có tín hiệu.

**Trạng thái**: pipeline 4 bước đã chạy xong, ĐÃ nâng cấp lên bản multi-lag +
dummy mùa vụ (đúng đặc tả Tetlock hơn). Kết quả: **vẫn không tìm thấy bằng
chứng vững chắc** về liên hệ sentiment ↔ lợi suất VN-Index cấp ngày (lag 1
không có ý nghĩa, kiểm định tổng lag2-5 "đảo chiều" không có ý nghĩa ở cả 2
cách; xem `Market_Validation/README.md` để biết chi tiết + 4 giới hạn còn
lại chưa khắc phục — đặc biệt điểm 1: inner-join làm méo nhẹ cấu trúc lag ở
~9,4% ngày). Chưa nên kết luận "phương pháp không có tín hiệu" — cần khắc
phục giới hạn 1 (dùng lịch giao dịch liên tục thay vì inner-join) và/hoặc
chia mẫu theo giai đoạn trước khi kết luận chắc chắn.

(Ghi chú: `News/Build_sentiment_index/build_sentiment_index_daily.py` đã có
sẵn từ trước — sửa lại đường dẫn input + generalize đọc 2 nguồn Lexicon_based/
thay vì tạo file mới trùng lặp trong `Market_Validation/`.)

---

## D. Nhánh Traditional_ML (điểm 5) — `Traditional_ML/`

- [ ] Thêm ComplementNB.
- [ ] Thêm kiểm định McNemar so 2 model.
- [ ] Báo cáo repeated CV kèm độ lệch chuẩn thay vì 1 con số đơn lẻ; chưa
      chốt Random Forest là lựa chọn cuối cùng (mô hình tuyến tính có thể
      vượt lên khi dữ liệu tăng).

**Trạng thái**: chưa động vào trong phiên làm việc này — cần đọc lại
`run_pipeline.py`/`verify_pipeline.py` (xem [[traditional-ml-pipeline]]) trước
khi sửa.

---

## E. Nhánh Transfer_Learning / PhoBERT (điểm 6) — `Transfer_Learning/`

Thứ tự mentor đề xuất (không nhảy thẳng vào fine-tune):

1. [ ] **Domain-adaptive pretraining**: tiếp tục pretrain PhoBERT bằng MLM
       trên 126k bài **chưa nhãn** — không cần ground truth, chạy nền được
       ngay.
2. [ ] **Frozen-feature extraction**: PhoBERT đóng băng trích đặc trưng +
       logistic regression trên top — ổn định hơn fine-tune toàn bộ khi còn
       ít nhãn.
3. [ ] Chỉ fine-tune toàn phần khi có vài trăm nhãn trở lên (chờ kết quả B).

**Trạng thái**: chưa bắt đầu.

---

## F. Chi tiết nhỏ (điểm 7) — `Lexicon_based/Scoring/classify_and_evaluate.py`

- [ ] Tách 2 trường hợp Neutral trong `assign_three_class_label()`:
      (a) `positive_score = negative_score = 0` vì bài **không match từ nào**
      trong dictionary, (b) hai điểm khác 0 nhưng bằng nhau — báo tỉ lệ riêng
      từng loại thay vì gộp chung "Neutral".

**Trạng thái**: việc nhỏ, làm nhanh, có thể làm kèm lúc sửa mục A.

---

## Thứ tự ưu tiên đề xuất

1. **B** (ground truth) — mentor gọi là ưu tiên số 1; cần thời gian + người,
   nên khởi động sớm và chạy song song nền trong lúc làm các mục khác.
2. **C** (kiểm định cấp-ngày) — không cần chờ B, tận dụng hạ tầng có sẵn,
   cho tín hiệu đáng tin ngay ở cỡ mẫu lớn.
3. **A** (sửa thống kê/tune-holdout) — nhanh, nên làm **trước khi báo cáo
   thêm bất kỳ con số accuracy nào mới** cho mentor.
4. **F** (chi tiết nhỏ) — làm tiện tay lúc sửa A.
5. **D, E** (ML / PhoBERT) — mentor xếp ưu tiên 2, làm khi rảnh tay từ nhánh
   Lexicon-Based.
