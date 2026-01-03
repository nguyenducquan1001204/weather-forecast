# 🚀 HƯỚNG DẪN SETUP GITHUB ACTIONS - TỪNG BƯỚC

## BƯỚC 1: Tạo Repository trên GitHub

1. Đăng nhập vào GitHub: https://github.com
2. Click nút **"+"** (góc trên bên phải) → **"New repository"**
3. Điền thông tin:
   - **Repository name**: `weather-forecast` (hoặc tên bạn muốn)
   - **Description**: "Weather forecasting with auto crawl and train"
   - **Public** ✅ (QUAN TRỌNG: Phải chọn Public để có 2000 phút/tháng free)
   - **Không** tích "Add a README file"
   - **Không** tích "Add .gitignore" (đã có rồi)
   - **Không** chọn license
4. Click **"Create repository"**

## BƯỚC 2: Push Code lên GitHub

Mở Terminal/PowerShell trong thư mục project và chạy:

```bash
# Kiểm tra xem đã có git chưa
git status

# Nếu chưa có git, khởi tạo
git init

# Thêm tất cả file
git add .

# Commit
git commit -m "Initial commit with GitHub Actions workflow"

# Thêm remote (thay YOUR_USERNAME và YOUR_REPO bằng tên thật)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push lên GitHub
git push -u origin main
```

**Lưu ý**: 
- Nếu repo của bạn dùng `master` thay vì `main`, đổi thành `git push -u origin master`
- Nếu GitHub yêu cầu đăng nhập, dùng Personal Access Token thay vì password

## BƯỚC 3: Kích hoạt GitHub Actions

1. Vào repository vừa tạo trên GitHub
2. Click tab **"Actions"** (ở trên cùng)
3. Nếu thấy thông báo **"Workflows aren't being run on this forked repository"** hoặc **"I understand my workflows, go ahead and enable them"**:
   - Click **"I understand my workflows, go ahead and enable them"**
4. Bạn sẽ thấy workflow **"Auto Crawl and Train"** trong danh sách

## BƯỚC 4: Test chạy thủ công (Tùy chọn)

1. Vào tab **"Actions"**
2. Click vào workflow **"Auto Crawl and Train"**
3. Click nút **"Run workflow"** (bên phải)
4. Chọn branch **"main"** (hoặc "master")
5. Click **"Run workflow"** (màu xanh)
6. Chờ vài phút, workflow sẽ chạy

## BƯỚC 5: Kiểm tra kết quả

1. Vào tab **"Actions"**
2. Click vào lần chạy mới nhất (có dấu tick xanh nếu thành công)
3. Xem logs để biết:
   - Crawl đã chạy chưa
   - Train đã chạy chưa
   - Database và models đã được commit chưa

## ⏰ LỊCH CHẠY TỰ ĐỘNG

- **Thời gian**: Mỗi ngày vào **3h sáng** (giờ Việt Nam)
- **Công việc**:
  1. Crawl dữ liệu thời tiết ngày hôm qua
  2. Train lại model với toàn bộ dữ liệu
  3. Tự động commit và push database + models

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Repository phải là PUBLIC
- Chỉ có **public repo** mới có 2000 phút/tháng free
- Private repo: 0 phút free (cần trả phí)

### 2. Database và Models sẽ được commit
- File `weather.db` và `weather_models_final.pkl` sẽ được push lên GitHub
- Nếu không muốn, sửa file `.gitignore`:
  ```
  *.db
  *.pkl
  ```

### 3. Thời gian chạy
- Crawl: ~5-10 phút
- Train: ~30-60 phút
- **Tổng**: ~35-70 phút/lần chạy
- Với 2000 phút/tháng, bạn có thể chạy ~28-57 lần/tháng

## 🔧 TROUBLESHOOTING

### Lỗi: "Workflows aren't being run"
- **Giải pháp**: Vào Settings → Actions → General → Enable workflows

### Lỗi: "Permission denied" khi push
- **Giải pháp**: Kiểm tra token có quyền write không
- Workflow đã tự động dùng `GITHUB_TOKEN`, không cần setup thêm

### Lỗi: Selenium/Chrome không chạy
- **Giải pháp**: Workflow đã tự động cài Chrome, kiểm tra logs để xem lỗi cụ thể

### Workflow không chạy tự động
- **Giải pháp**: 
  1. Kiểm tra repository có public không
  2. Kiểm tra cron schedule có đúng không
  3. GitHub Actions có thể delay vài phút

## 📝 TÙY CHỈNH

### Thay đổi thời gian chạy

Sửa file `.github/workflows/auto_crawl_train.yml`:

```yaml
schedule:
  # 3h sáng VN = 20:00 UTC
  - cron: '0 20 * * *'
  
  # 6h sáng VN = 23:00 UTC
  # - cron: '0 23 * * *'
```

### Chỉ crawl, không train

Comment bước train:

```yaml
# - name: Train models with all data
#   run: |
#     python train_final_models.py
```

## ✅ KIỂM TRA THÀNH CÔNG

Sau khi setup xong, bạn sẽ thấy:

1. ✅ Workflow chạy thành công trong tab **Actions**
2. ✅ File `weather.db` và `weather_models_final.pkl` được commit mỗi ngày
3. ✅ Logs hiển thị: "Crawl completed", "Train completed"

---

**Chúc bạn setup thành công! 🎉**

