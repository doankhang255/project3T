# Nguồn xác thực — từ nhấn mạnh & từ phủ định tiếng Việt

Tài liệu này đánh giá **độ tin cậy thực sự** của 2 danh sách đang dùng trong
pipeline: `intensifier_words.txt` (thư mục này) và `negation_cue_words.txt`
(`Seed_set_Prepare/negation_cue_words.txt`). Xếp hạng nguồn theo 3 mức:

- **Học thuật** (bài báo/tạp chí ngôn ngữ học có phản biện) — độ tin cậy cao nhất.
- **Chính thống phi học thuật** (chương trình giáo dục Bộ GD&ĐT, sách giáo khoa) — đáng tin nhưng không phải nghiên cứu độc lập.
- **Tham khảo phổ thông** (trang học tiếng Việt/luyện thi, blog giáo dục) — chỉ dùng để đối chiếu, không phải căn cứ chính.

---

## 1. Từ phủ định (`negation_cue_words.txt`) — ĐÃ kiểm tra lại

**Nhóm 1 (cốt lõi): không, chẳng, chả, đâu/đâu_có, nào_có, khỏi, cóc, đếch, ứ, chưa**
→ Nguồn **học thuật**, xác nhận trực tiếp:

> "FUNCTION WORDS OF NEGATION IN VIETNAMESE", *Tạp chí Ngôn ngữ và Đời sống* (peer-reviewed).
> https://vjol.info.vn/index.php/NNDS/article/view/19433

Bài báo liệt kê đúng 10 từ trên là "function words of negation" tiếng Việt,
và có 1 chi tiết quan trọng: **chỉ "không" mới xuất hiện được ở mọi phong
cách chức năng ngôn ngữ** (kể cả văn phong hành chính/báo chí formal); các từ
còn lại (khỏi, cóc, đếch, ứ...) *"usually appear in art, living activities or
deliberate writing"* — tức **chủ yếu khẩu ngữ/văn nghệ, hiếm gặp trong tin
tài chính**. Đây là căn cứ học thuật xác nhận quyết định trước đó của tôi:
xếp nhóm khẩu ngữ (khỏi, cóc, đếch, ứ) vào diện "có thể bỏ nếu muốn gọn".

**Nhóm 2-6 (không_hề, chưa_từng, không_thể, đâu_có_phải...)**
→ **KHÔNG tìm được 1 nguồn học thuật liệt kê nguyên cụm** này. Đây là các tổ
hợp tôi suy luận hợp lý bằng cách ghép từ phủ định gốc (đã xác nhận ở Nhóm 1)
với các phó từ/cụm tăng cường thông dụng (hề, bao_giờ, còn, từng, thể...) —
về mặt cấu trúc câu tiếng Việt đây là cách ghép chuẩn, nhưng **chưa có nguồn
độc lập xác nhận từng cụm cụ thể**. Cần bạn tự đối chiếu bằng cảm nhận người
bản ngữ, hoặc tôi có thể thử tìm thêm nếu cần độ chắc chắn cao hơn.

**Kết luận**: nhóm 1 đáng tin cậy cao (học thuật). Nhóm 2-6 hợp lý về ngữ
pháp nhưng là suy luận, không phải trích dẫn — nên coi là "cần review thêm"
chứ không phải "đã xác thực".

---

## 2. Từ nhấn mạnh (`intensifier_words.txt`) — MỚI xây dựng

**Nhóm 1-2 (hơi, khá, rất, quá, lắm, thật...)**
→ Nguồn **chính thống phi học thuật**: đây là danh sách "phó từ chỉ mức độ"
trong chương trình Ngữ văn 6-7 (bộ sách "Kết nối tri thức") — phân loại ngữ
pháp CHÍNH THỨC do Bộ Giáo dục và Đào tạo quy định, không phải blog tự chế.
Nguồn tham khảo: vndoc.com, rdsic.edu.vn, tech12h.com (đều trích dẫn cùng
nội dung sách giáo khoa). **Không tìm được bài báo ngôn ngữ học riêng** về
chủ đề này bằng tìm kiếm web — khác với từ phủ định, "phó từ chỉ mức độ"
dường như ít là đề tài nghiên cứu học thuật riêng (có thể vì được coi là
kiến thức ngữ pháp cơ bản, ít tranh cãi).

**Nhóm 3 (cực_kỳ, vô_cùng, hết_sức, tuyệt_đối, hoàn_toàn, đặc_biệt)**
→ Xuất hiện lặp lại trong nhiều nguồn tham khảo phổ thông cùng nhóm 1-2,
nhưng bản thân các nguồn đó cũng chỉ là trang học tiếng Việt, không phải
SGK/học thuật trực tiếp. Độ tin cậy: **trung bình**.

**Nhóm 4 (đáng_kể, nghiêm_trọng, mạnh_mẽ, sâu_sắc, rõ_rệt)**
→ Đây là **tôi tự thêm dựa trên quan sát văn phong báo chí tài chính**
trong quá trình review seed suốt phiên làm việc này (các từ này hay xuất
hiện làm bổ ngữ tăng cường mức độ, ví dụ "giảm nghiêm_trọng", "tăng rõ_rệt").
**KHÔNG có nguồn ngôn ngữ học xác nhận** đây là "phó từ chỉ mức độ" theo
đúng nghĩa ngữ pháp — về bản chất chúng là tính từ/trạng từ mượn dùng như bổ
ngữ mức độ trong văn phong chuyên ngành, khác bản chất với nhóm 1-3. **Cần
bạn tự quyết định có giữ nhóm này không**, hoặc tôi kiểm tra corpus thực tế
xem tần suất xuất hiện đúng vai trò "nhấn mạnh" trước khi giữ.

**Kết luận**: nhóm 1-2 có căn cứ chính thống (chương trình giáo dục quốc
gia) nhưng thấp hơn 1 bậc so với nguồn học thuật của từ phủ định. Nhóm 3
trung bình. Nhóm 4 là suy luận riêng của tôi, độ tin cậy thấp nhất, nên xử
lý thận trọng nhất (khuyến nghị: kiểm tra corpus trước khi dùng, hoặc bỏ).

---

## Tóm tắt độ tin cậy

| Danh sách | Nhóm | Độ tin cậy | Nguồn |
|---|---|---|---|
| Phủ định | Nhóm 1 (10 từ gốc) | **Cao** | Bài báo học thuật (Tạp chí Ngôn ngữ và Đời sống) |
| Phủ định | Nhóm 2-6 (tổ hợp) | Trung bình | Suy luận ngữ pháp, chưa có trích dẫn riêng |
| Nhấn mạnh | Nhóm 1-2 | Trung bình-cao | Chương trình Ngữ văn 6-7 (Bộ GD&ĐT) |
| Nhấn mạnh | Nhóm 3 | Trung bình | Lặp lại ở nhiều nguồn phổ thông, không học thuật |
| Nhấn mạnh | Nhóm 4 | **Thấp** | Tự quan sát corpus, chưa xác thực |
