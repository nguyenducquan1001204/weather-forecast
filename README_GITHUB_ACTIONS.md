# Hướng dẫn Setup GitHub Actions - Tự động Crawl và Train

## Yêu cầu

1. Repository phải là **public** (để có 2000 phút/tháng free)
2. Hoặc repository private với GitHub Pro/Team (có 3000 phút/tháng)

## Các bước setup

### 1. Push code lên GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Kiểm tra workflow file

Đảm bảo file `.github/workflows/auto_crawl_train.yml` đã được tạo và commit.

### 3. Kích hoạt GitHub Actions

1. Vào repository trên GitHub
2. Vào tab **Actions**
3. Nếu lần đầu, click **"I understand my workflows, go ahead and enable them"**
4. Workflow sẽ tự động chạy vào 3h sáng mỗi ngày (theo giờ VN)

### 4. Chạy thủ công (test)

1. Vào tab **Actions**
2. Chọn workflow **"Auto Crawl and Train"**
3. Click **"Run workflow"** → **"Run workflow"**

## Lịch chạy tự động

- **Thời gian**: Mỗi ngày vào **3h sáng** (giờ Việt Nam)
- **Công việc**:
  1. Crawl dữ liệu thời tiết ngày hôm qua
  2. Train lại model với toàn bộ dữ liệu
  3. Commit và push database + models lên GitHub

## Lưu ý

### Database và Models

- File `weather.db` và `weather_models_final.pkl` sẽ được commit và push lên GitHub
- Nếu không muốn commit, sửa file `.gitignore`:
  ```
  *.db
  *.pkl
  ```

### Quota GitHub Actions

- **Public repo**: 2000 phút/tháng (≈ 66 phút/ngày)
- **Private repo**: 0 phút free (cần GitHub Pro/Team)

### Thời gian chạy

- Crawl: ~5-10 phút
- Train: ~30-60 phút (tùy dữ liệu)
- **Tổng**: ~35-70 phút/lần chạy

### Lưu ý về Selenium

- GitHub Actions sử dụng Ubuntu Linux
- Chrome/ChromeDriver được cài đặt tự động trong workflow
- Code đã được cập nhật để hỗ trợ Linux

## Troubleshooting

### Workflow không chạy

1. Kiểm tra repository có public không
2. Kiểm tra tab **Actions** đã được enable chưa
3. Kiểm tra file workflow có đúng syntax không

### Lỗi Selenium/Chrome

- Workflow đã tự động cài Chrome và ChromeDriver
- Nếu vẫn lỗi, kiểm tra logs trong tab **Actions**

### Lỗi commit/push

- Đảm bảo workflow có quyền write
- Kiểm tra `GITHUB_TOKEN` có sẵn (tự động có)

## Tùy chỉnh

### Thay đổi thời gian chạy

Sửa file `.github/workflows/auto_crawl_train.yml`:

```yaml
schedule:
  # Cron format: phút giờ ngày tháng thứ
  # 3h sáng VN = 20:00 UTC (ngày hôm trước)
  - cron: '0 20 * * *'  # 3h sáng VN
  # Hoặc 6h sáng VN = 23:00 UTC
  - cron: '0 23 * * *'  # 6h sáng VN
```

### Chỉ crawl, không train

Comment bước train trong workflow:

```yaml
# - name: Train models with all data
#   run: |
#     python train_final_models.py
```

### Chỉ train, không crawl

Comment bước crawl trong workflow:

```yaml
# - name: Crawl weather data
#   run: |
#     python crawl_weather.py
```

## Kiểm tra kết quả

1. Vào tab **Actions** → Xem logs của mỗi lần chạy
2. Kiểm tra file `weather.db` và `weather_models_final.pkl` có được update không
3. Xem commit history để biết lần chạy gần nhất

