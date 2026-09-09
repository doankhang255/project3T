# Deploy Label Studio public trên Render (free) + Neon Postgres

Kết quả: `https://ground-truth-sentiment.onrender.com`, $0/tháng, dữ liệu nằm ở
Neon nên không mất. Đổi lại: ngủ sau 15' không dùng → lần mở kế tiếp chờ ~1'.

Render free **không có persistent disk** → bắt buộc dùng DB ngoài (Neon). Ta đã
có sẵn.

---

## 1. Tạo tài khoản Render

https://render.com → Sign up (Google/GitHub/email). Web service free **không
cần thẻ**.

## 2. New Web Service từ Docker image

1. Dashboard → **New +** → **Web Service**.
2. Chọn **Deploy an existing image from a registry** (không phải "Build from a
   repository").
3. **Image URL**: `docker.io/heartexlabs/label-studio:1.23.0`
4. **Next**.

## 3. Cấu hình service

| Trường | Giá trị |
|---|---|
| **Name** | `ground-truth-sentiment` (→ URL `https://ground-truth-sentiment.onrender.com`) |
| **Region** | Singapore |
| **Instance Type** | **Free** |
| **Health Check Path** | để trống |

## 4. Environment Variables

Kéo xuống mục **Environment Variables** → nếu có nút **Add from .env** thì dán
nguyên khối dưới; không thì thêm từng dòng.

```
PORT=8080
WEB_CONCURRENCY=1
LABEL_STUDIO_HOST=https://ground-truth-sentiment.onrender.com
CSRF_TRUSTED_ORIGINS=https://ground-truth-sentiment.onrender.com
DJANGO_DB=default
STORAGE_PERSISTENCE=1
LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
POSTGRE_HOST=ep-withered-credit-b3r2dfz3.c-4.ap-southeast-1.aws.neon.tech
POSTGRE_PORT=5432
POSTGRE_NAME=neondb
POSTGRE_USER=neondb_owner
LABEL_STUDIO_USERNAME=doankhangll255@gmail.com
SECRET_KEY=9KodgCMML_jbjhnyI5lzfpep776_Qo7VuAbcyPtRxo0o0GCNyai9PlCgocgrFKPqCo0
```

Thêm **2 biến nhạy cảm** riêng (đừng để lộ):

| Key | Value |
|---|---|
| `POSTGRE_PASSWORD` | phần `npg_...` trong connection string Neon |
| `LABEL_STUDIO_PASSWORD` | mật khẩu admin mạnh do bạn đặt, nhớ kỹ |

> Nếu Render cấp URL khác (do tên bị trùng, nó thêm hậu tố), sau khi tạo xong
> vào tab **Environment**, sửa lại `LABEL_STUDIO_HOST` và `CSRF_TRUSTED_ORIGINS`
> cho khớp URL thật rồi **Manual Deploy → Deploy latest**.

## 5. Create Web Service

Bấm **Create Web Service**. Xem tab **Logs**:

- Lần đầu chạy migration lên Neon (~30–60s).
- Chờ tới khi thấy dòng kiểu `Starting gunicorn` / `Listening at: http://0.0.0.0:8080`
  và trạng thái service chuyển **Live**.

## 6. Đăng nhập + kiểm tra khóa đăng ký

1. Mở `https://ground-truth-sentiment.onrender.com` (lần đầu chờ ~1').
2. Login bằng `LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD`.
3. Mở cửa sổ ẩn danh → vào lại link → **không được** có nút Sign Up.

## 7. (Tùy chọn) Giữ cho đỡ ngủ

https://uptimerobot.com → New Monitor → HTTP(s) → URL của bạn → interval 5–10'.
Giữ 24/7 ăn ~730/750 giờ free mỗi tháng → chỉ chạy đúng **1 service** này thôi.
An toàn hơn: đặt monitor chỉ chạy khung 7h–24h.

## 8. Tạo project + import bài

1. **Create Project** → `ground_truth_sentiment`.
2. **Settings → Labeling Interface → Code** → dán nội dung
   `hf_space/label_config.xml` (trường dữ liệu là `content`) → Save.
3. **Import** → kéo file CSV/JSON có cột `content`.
4. Muốn 2 người gán chồng 1 bài: **Settings → Annotation → Overlap of
   annotations** = 2.

## 9. Mời annotator

Menu user → **Organization** → **Add People** → copy invite link, gửi riêng
từng người. Mỗi người 1 tài khoản riêng (để tính Cohen's kappa — xem
`../News/Build_sentiment_label/MENTOR_FEEDBACK_PLAN.md` mục B).

## 10. Nối lại với code repo

`Label_Studio/label_studio_client.py` đọc `.env`:

```
LABEL_STUDIO_URL=https://ground-truth-sentiment.onrender.com
LABEL_STUDIO_TOKEN=<Access Token trong Account & Settings của app>
```

---

## Sau khi chạy ổn — đổi password Neon

Password `npg_...` đã xuất hiện trong lịch sử chat. Vào Neon → **Roles** →
`neondb_owner` → **Reset password** → cập nhật lại `POSTGRE_PASSWORD` trên
Render → Deploy latest.

## Troubleshooting

| Triệu chứng | Xử lý |
|---|---|
| Service cứ **restart loop**, logs có `Killed` / OOM | 512MB hơi thiếu. Đảm bảo `WEB_CONCURRENCY=1`. Nếu vẫn chết → chuyển Koyeb, hoặc Fly.io máy 1GB (~$3/tháng). |
| Logs: `could not connect` / timeout tới Postgres | Sai `POSTGRE_HOST` (đang dùng bản `-pooler`?). Dùng host trực tiếp. |
| Logs: `SSL connection is required` | Hiếm với LS 1.23 (psycopg2 mặc định `sslmode=prefer`). Nếu gặp: thêm biến `PGSSLMODE=require`. |
| Vào link bị `Bad Request (400)` / CSRF khi login | `LABEL_STUDIO_HOST` / `CSRF_TRUSTED_ORIGINS` chưa khớp URL thật. Sửa cho đúng, Deploy latest. |
| Mất project sau khi service restart | `DJANGO_DB`/`POSTGRE_*` sai → LS rơi về SQLite ephemeral. Soát lại mục 4. |
