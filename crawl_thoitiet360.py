"""
Script crawl và tiền xử lý dữ liệu dự báo thời tiết từ thoitiet360.edu.vn
Để so sánh với dự đoán của hệ thống
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import subprocess
import os
from datetime import datetime, timedelta
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Lấy thư mục nơi script này được đặt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_pull_from_github():
    """Kiểm tra và pull cập nhật từ GitHub nếu có"""
    print("="*70)
    print("KIEM TRA CAP NHAT TU GITHUB")
    print("="*70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đang kiểm tra cập nhật...")
    print(f"  Thư mục làm việc: {SCRIPT_DIR}")
    
    try:
        # Chuyển đến thư mục script để đảm bảo đúng vị trí
        os.chdir(SCRIPT_DIR)
        
        # Lấy các thay đổi mới nhất
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Không thể fetch từ GitHub (có thể không phải git repo): {result.stderr[:100]}")
            return False
        
        # Kiểm tra xem có commit mới không
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD..origin/main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Không thể kiểm tra commits: {result.stderr[:100]}")
            return False
        
        commits_behind = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        if commits_behind > 0:
            print(f"  📥 Tìm thấy {commits_behind} commit(s) mới. Đang pull cập nhật...")
            
            # Pull các thay đổi
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"  ✅ Đã pull thành công {commits_behind} commit(s)")
                print(f"  📄 Các file đã cập nhật: thoitiet360_data.csv, database, và các file khác")
                return True
            else:
                print(f"  ⚠️  Lỗi khi pull: {result.stderr[:100]}")
                return False
        else:
            print("  ✅ Đã cập nhật mới nhất, không có thay đổi")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Lỗi khi kiểm tra GitHub: {str(e)[:100]}")
        return False

# Mapping thành phố
CITY_MAPPING = {
    'ha-noi': 'Hà Nội',
    'vinh': 'Vinh',
    'ho-chi-minh': 'Hồ Chí Minh'
}

# URL mapping
CITY_URLS = {
    'ha-noi': 'https://thoitiet360.edu.vn/ha-noi/3-ngay-toi',
    'vinh': 'https://thoitiet360.edu.vn/nghe-an/vinh/3-ngay-toi',
    'ho-chi-minh': 'https://thoitiet360.edu.vn/ho-chi-minh/3-ngay-toi'
}

def parse_temperature(temp_str):
    """Parse nhiệt độ từ string (ví dụ: "14°" -> 14.0)"""
    if not temp_str:
        return None
    try:
        # Loại bỏ ký tự ° và khoảng trắng
        temp_str = temp_str.replace('°', '').replace('°C', '').strip()
        return float(temp_str)
    except:
        return None

def parse_pressure(pressure_str):
    """Parse áp suất từ string (ví dụ: "1028 hPa" -> 1028.0)"""
    if not pressure_str:
        return None
    try:
        # Loại bỏ "hPa" và khoảng trắng
        pressure_str = pressure_str.replace('hPa', '').strip()
        return float(pressure_str)
    except:
        return None

def parse_wind(wind_str):
    """Parse gió từ string (ví dụ: "6.92 km/h" -> 6.92)"""
    if not wind_str:
        return None
    try:
        # Loại bỏ "km/h" và khoảng trắng
        wind_str = wind_str.replace('km/h', '').strip()
        return float(wind_str)
    except:
        return None

def parse_rain(rain_str):
    """Parse lượng mưa từ string (ví dụ: "0 mm" -> 0.0)"""
    if not rain_str:
        return None
    try:
        # Loại bỏ "mm" và khoảng trắng
        rain_str = rain_str.replace('mm', '').strip()
        return float(rain_str)
    except:
        return None

def parse_cloud(cloud_str):
    """Parse mây từ string (có thể là text mô tả)"""
    # Thoitiet360 có thể không có % mây, chỉ có mô tả
    # Trả về None nếu không parse được
    return None

def crawl_thoitiet360(city_key='ha-noi'):
    """
    Crawl dữ liệu dự báo ngày hôm nay từ thoitiet360.edu.vn
    
    Args:
        city_key: 'ha-noi', 'vinh', hoặc 'ho-chi-minh'
    
    Returns:
        List of dicts với dữ liệu dự báo ngày hôm nay
    """
    url = CITY_URLS.get(city_key)
    if not url:
        print(f"⚠️  Không tìm thấy URL cho thành phố: {city_key}")
        return []
    
    print(f"\n🔍 Đang crawl dữ liệu từ thoitiet360.edu.vn cho {CITY_MAPPING.get(city_key, city_key)}...")
    print(f"   URL: {url}")
    
    # Retry logic: thử lại tối đa 3 lần
    max_retries = 3
    retry_delay = 2  # Nghỉ 2 giây giữa các lần thử
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive'
            }
            
            # Sử dụng session để giữ kết nối
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=30)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  HTTP {response.status_code}, thử lại sau {retry_delay} giây...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ Lỗi: HTTP {response.status_code} sau {max_retries} lần thử")
                    return []
            
            # Thành công, thoát khỏi vòng lặp retry
            break
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  Lỗi kết nối: {str(e)[:50]}...")
                print(f"   ⚠️  Thử lại lần {attempt + 2}/{max_retries} sau {retry_delay} giây...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Lỗi kết nối sau {max_retries} lần thử: {str(e)[:100]}")
                return []
        except Exception as e:
            print(f"❌ Lỗi không xác định: {str(e)[:100]}")
            return []
    
    try:
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        import re
        forecast_data = []
        today = datetime.now().date()
        
        # Tìm tất cả các heading có chứa pattern ngày (T2-T7, CN, hoặc số ngày/tháng)
        date_pattern = re.compile(r'(T[2-7]|CN|Chủ nhật|Thứ [2-7])\s*\d{1,2}/\d{1,2}', re.IGNORECASE)
        
        found_days = []
        
        # Tìm tất cả các phần tử có chứa pattern ngày
        all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section', 'article'])
        
        for element in all_elements:
            text = element.get_text()
            
            # Kiểm tra xem có chứa pattern ngày không
            if date_pattern.search(text):
                # Tìm nhiệt độ trong phần tử này (lấy số đầu tiên hợp lý)
                temp_matches = re.findall(r'(\d+)\s*°', text)
                main_temp = None
                if temp_matches:
                    for temp in temp_matches:
                        temp_val = int(temp)
                        # Nhiệt độ hợp lý cho Việt Nam: 0-50°C
                        if 0 <= temp_val <= 50:
                            main_temp = temp
                            break
                
                if main_temp:
                    # Tìm các thông số khác
                    pressure_matches = re.findall(r'(\d{3,4})\s*hPa', text)
                    wind_matches = re.findall(r'(\d+\.?\d*)\s*km/h', text)
                    rain_matches = re.findall(r'(\d+\.?\d*)\s*mm', text)
                    
                    # Kiểm tra xem đã có ngày này chưa (tránh trùng lặp)
                    day_key = f"{main_temp}_{pressure_matches[0] if pressure_matches else 'none'}"
                    if day_key not in [d.get('key', '') for d in found_days]:
                        found_days.append({
                            'key': day_key,
                            'temp': main_temp,
                            'pressure': pressure_matches[0] if pressure_matches else None,
                            'wind': wind_matches[0] if wind_matches else None,
                            'rain': rain_matches[0] if rain_matches else None,
                            'text': text[:200]
                        })
        
        # Chỉ lấy ngày hôm nay (ngày đầu tiên)
        found_days = found_days[:1]
        
        print(f"   Tìm thấy {len(found_days)} ngày dự báo (chỉ lấy hôm nay)")
        
        # Tạo record cho ngày hôm nay
        for idx, day_data in enumerate(found_days):
            forecast_date = today  # Chỉ lấy ngày hôm nay
            
            # Làm sạch raw_text: thay thế ký tự xuống dòng bằng khoảng trắng
            raw_text = day_data.get('text', '')
            raw_text = raw_text.replace('\n', ' ').replace('\r', ' ').strip()
            raw_text = ' '.join(raw_text.split())  # Loại bỏ khoảng trắng thừa
            raw_text = raw_text[:100]  # Giới hạn 100 ký tự
            
            record = {
                'city': CITY_MAPPING.get(city_key, city_key),
                'city_key': city_key,
                'date': forecast_date.strftime('%Y-%m-%d'),
                'source': 'thoitiet360',
                'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Temp': parse_temperature(day_data['temp']) if day_data['temp'] else None,
                'Pressure': parse_pressure(day_data['pressure']) if day_data['pressure'] else None,
                'Wind': parse_wind(day_data['wind']) if day_data['wind'] else None,
                'Rain': parse_rain(day_data['rain']) if day_data['rain'] else None,
                'Cloud': None,
                'raw_text': raw_text
            }
            
            forecast_data.append(record)
            print(f"   ✓ Ngày {forecast_date.strftime('%Y-%m-%d')}: Temp={record['Temp']}°C, Pressure={record['Pressure']}hPa, Wind={record['Wind']}km/h, Rain={record['Rain']}mm")
        
        # Nếu không parse được bằng cách trên, thử cách khác
        if not forecast_data:
            print("   ⚠️  Không parse được dữ liệu bằng cách thông thường, thử cách khác...")
            
            # In ra một phần HTML để debug
            print(f"   HTML sample (first 1000 chars): {response.text[:1000]}")
        
        return forecast_data
        
    except Exception as e:
        print(f"❌ Lỗi khi parse dữ liệu: {str(e)[:100]}")
        return []

def preprocess_thoitiet360_data(df):
    """
    Tiền xử lý dữ liệu từ thoitiet360 để so sánh (không cần các cột cho training)
    
    Args:
        df: DataFrame từ crawl_thoitiet360
    
    Returns:
        DataFrame đã được tiền xử lý, chỉ giữ các cột cần thiết để so sánh
    """
    df = df.copy()
    
    # 1. Mapping tên thành phố sang format database
    city_mapping = {
        'Hà Nội': 'ha-noi',
        'Vinh': 'vinh',
        'Hồ Chí Minh': 'ho-chi-minh-city'
    }
    
    if 'city' in df.columns:
        df['city'] = df['city'].map(city_mapping).fillna(df['city'])
    
    # 2. Tạo datetime từ date (mặc định 00:00:00)
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' 00:00:00')
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 3. Đảm bảo các cột số là numeric
    numeric_cols = ['Temp', 'Pressure', 'Wind', 'Rain', 'Cloud', 'Gust']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = None
    
    # 4. Chỉ giữ lại các cột cần thiết để so sánh
    columns_to_keep = ['city', 'date', 'datetime', 'Temp', 'Pressure', 'Wind', 'Rain', 'Cloud', 'Gust']
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    # 5. Loại bỏ các cột không cần thiết
    columns_to_drop = ['raw_text', 'city_key', 'source', 'crawled_at', 'Time']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 6. Sắp xếp theo city và date
    if 'city' in df.columns and 'date' in df.columns:
        df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    return df

def save_to_csv(data, filename='thoitiet360_data.csv'):
    """Lưu dữ liệu gốc vào file CSV"""
    try:
        if not data:
            print("⚠️  Không có dữ liệu để lưu")
            return
        
        # Lưu file gốc (chưa tiền xử lý)
        df_raw = pd.DataFrame(data)
        if 'raw_text' in df_raw.columns:
            df_raw['raw_text'] = df_raw['raw_text'].astype(str).str.replace('\n', ' ').str.replace('\r', ' ').str.strip()
            df_raw['raw_text'] = df_raw['raw_text'].str[:100]
        
        df_raw.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(df_raw)} records vào {filename}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu CSV: {str(e)}")

def save_to_database(df):
    """Lưu dữ liệu vào database"""
    try:
        from database import init_database, get_db_connection
        import os
        
        # Khởi tạo database
        init_database()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        inserted = 0
        updated = 0
        
        for _, row in df.iterrows():
            # Kiểm tra xem đã có record chưa (dựa trên city và date)
            cursor.execute('''
                SELECT id FROM thoitiet360_data 
                WHERE city = ? AND date = ?
            ''', (row['city'], row['date']))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update record đã tồn tại
                cursor.execute('''
                    UPDATE thoitiet360_data 
                    SET datetime = ?, Temp = ?, Pressure = ?, Wind = ?, Rain = ?, Cloud = ?, Gust = ?
                    WHERE city = ? AND date = ?
                ''', (
                    row.get('datetime'),
                    row.get('Temp'),
                    row.get('Pressure'),
                    row.get('Wind'),
                    row.get('Rain'),
                    row.get('Cloud'),
                    row.get('Gust'),
                    row['city'],
                    row['date']
                ))
                updated += 1
            else:
                # Insert record mới
                cursor.execute('''
                    INSERT INTO thoitiet360_data 
                    (city, date, datetime, Temp, Pressure, Wind, Rain, Cloud, Gust)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['city'],
                    row['date'],
                    row.get('datetime'),
                    row.get('Temp'),
                    row.get('Pressure'),
                    row.get('Wind'),
                    row.get('Rain'),
                    row.get('Cloud'),
                    row.get('Gust')
                ))
                inserted += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Đã lưu vào database: {inserted} records mới, {updated} records cập nhật")
        return inserted + updated
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu vào database: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Crawl dữ liệu cho tất cả các thành phố"""
    # Kiểm tra và pull cập nhật từ GitHub trước
    check_and_pull_from_github()
    
    print("\n" + "="*70)
    print("CRAWL DỮ LIỆU NGÀY HÔM NAY TỪ THOITIET360.EDU.VN")
    print("="*70)
    
    all_data = []
    
    cities = ['ha-noi', 'vinh', 'ho-chi-minh']
    
    for idx, city_key in enumerate(cities, 1):
        print(f"\n[{idx}/{len(cities)}] Đang crawl {CITY_MAPPING.get(city_key, city_key)}...")
        data = crawl_thoitiet360(city_key)
        all_data.extend(data)
        
        # Nghỉ giữa các request để tránh bị chặn
        if idx < len(cities):
            time.sleep(3)  # Nghỉ 3 giây giữa các thành phố
    
    print(f"\n{'='*70}")
    print(f"TỔNG KẾT: Crawl được {len(all_data)} records")
    print(f"{'='*70}")
    
    if all_data:
        # Tiền xử lý dữ liệu
        df = pd.DataFrame(all_data)
        df_processed = preprocess_thoitiet360_data(df)
        
        # Lưu vào CSV (file gốc)
        save_to_csv(all_data, 'thoitiet360_data.csv')
        
        # Lưu vào database
        print("\n💾 Đang lưu vào database...")
        save_to_database(df_processed)
        
        # Hiển thị summary
        print("\n📊 Tóm tắt dữ liệu:")
        if not df.empty:
            print(df.groupby('city').size())
            print("\nMẫu dữ liệu đã lưu:")
            print(df_processed[['city', 'date', 'Temp', 'Pressure', 'Wind', 'Rain']].head(10).to_string())
    else:
        print("⚠️  Không crawl được dữ liệu nào!")

if __name__ == '__main__':
    main()

