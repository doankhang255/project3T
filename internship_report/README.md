# Internship Semester Final Report

Đây là bộ template Quarto để viết và render báo cáo cuối kỳ thực tập dạng PDF.

## Cách render

```powershell
cd C:\Users\doank\Documents\dev\project3T\internship_report
quarto render final_report.qmd --to pdf
```

File PDF sẽ được xuất vào thư mục `_output`.

Nếu máy báo thiếu LaTeX/TinyTeX, chạy:

```powershell
quarto install tinytex
```

## Các phần cần sửa trước khi nộp

- Thay thông tin ở trang bìa: trường, khoa, tên sinh viên, MSSV, lớp, công ty, giảng viên hướng dẫn, người hướng dẫn tại doanh nghiệp.
- Cập nhật thời gian thực tập và mô tả doanh nghiệp.
- Bổ sung số liệu thật ở Chương 4 và Chương 5.
- Thêm hình, bảng kết quả, biểu đồ EDA hoặc performance nếu đã có.
