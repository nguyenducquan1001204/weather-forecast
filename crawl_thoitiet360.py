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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_pull_from_github():
    print("="*70)
    print("KIEM TRA CAP NHAT TU GITHUB")
    print("="*70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đang kiểm tra cập nhật...")
    print(f"  Thư mục làm việc: {SCRIPT_DIR}")
    
    try:
        os.chdir(SCRIPT_DIR)
        
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Không thể fetch từ GitHub (có thể không phải git repo): {result.stderr[:100]}")
            return False
        
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

CITY_MAPPING = {
    'ha-noi': 'Hà Nội',
    'vinh': 'Vinh',
    'ho-chi-minh': 'Hồ Chí Minh'
}

CITY_URLS = {
    'ha-noi': 'https://thoitiet360.edu.vn/ha-noi/3-ngay-toi',
    'vinh': 'https://thoitiet360.edu.vn/nghe-an/vinh/3-ngay-toi',
    'ho-chi-minh': 'https://thoitiet360.edu.vn/ho-chi-minh/3-ngay-toi'
}

def parse_temperature(temp_str):
    if not temp_str:
        return None
    try:
        temp_str = temp_str.replace('°', '').replace('°C', '').strip()
        return float(temp_str)
    except:
        return None

def parse_pressure(pressure_str):
    if not pressure_str:
        return None
    try:
        pressure_str = pressure_str.replace('hPa', '').strip()
        return float(pressure_str)
    except:
        return None

def parse_wind(wind_str):
    if not wind_str:
        return None
    try:
        wind_str = wind_str.replace('km/h', '').strip()
        return float(wind_str)
    except:
        return None

def parse_rain(rain_str):
    if not rain_str:
        return None
    try:
        rain_str = rain_str.replace('mm', '').strip()
        return float(rain_str)
    except:
        return None

def parse_cloud(cloud_str):
    return None

def crawl_thoitiet360(city_key='ha-noi'):
    url = CITY_URLS.get(city_key)
    if not url:
        print(f"⚠️  Không tìm thấy URL cho thành phố: {city_key}")
        return []
    
    print(f"\n🔍 Đang crawl dữ liệu từ thoitiet360.edu.vn cho {CITY_MAPPING.get(city_key, city_key)}...")
    print(f"   URL: {url}")
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive'
            }
            
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
        
        date_pattern = re.compile(r'(T[2-7]|CN|Chủ nhật|Thứ [2-7])\s*\d{1,2}/\d{1,2}', re.IGNORECASE)
        
        found_days = []
        
        all_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'section', 'article'])
        
        for element in all_elements:
            text = element.get_text()
            
            if date_pattern.search(text):
                temp_matches = re.findall(r'(\d+)\s*°', text)
                main_temp = None
                if temp_matches:
                    for temp in temp_matches:
                        temp_val = int(temp)
                        if 0 <= temp_val <= 50:
                            main_temp = temp
                            break
                
                if main_temp:
                    pressure_matches = re.findall(r'(\d{3,4})\s*hPa', text)
                    wind_matches = re.findall(r'(\d+\.?\d*)\s*km/h', text)
                    rain_matches = re.findall(r'(\d+\.?\d*)\s*mm', text)
                    
                    temp_min_max_pattern = re.findall(r'(\d+\.?\d*)\s*°\s*/\s*(\d+\.?\d*)\s*°', text)
                    temp_min = None
                    temp_max = None
                    if temp_min_max_pattern:
                        temp_min = temp_min_max_pattern[0][0]
                        temp_max = temp_min_max_pattern[0][1]
                    else:
                        temp_min_max_pattern2 = re.findall(r'Thấp[^:]*:\s*(\d+\.?\d*)\s*°[^/]*/\s*Cao[^:]*:\s*(\d+\.?\d*)\s*°', text, re.IGNORECASE)
                        if temp_min_max_pattern2:
                            temp_min = temp_min_max_pattern2[0][0]
                            temp_max = temp_min_max_pattern2[0][1]
                        else:
                            parent = element.find_parent()
                            if parent:
                                parent_text = parent.get_text()
                                temp_min_max_pattern3 = re.findall(r'(\d+\.?\d*)\s*°\s*/\s*(\d+\.?\d*)\s*°', parent_text)
                                if temp_min_max_pattern3:
                                    temp_min = temp_min_max_pattern3[0][0]
                                    temp_max = temp_min_max_pattern3[0][1]
                    
                    day_key = f"{main_temp}_{pressure_matches[0] if pressure_matches else 'none'}"
                    if day_key not in [d.get('key', '') for d in found_days]:
                        found_days.append({
                            'key': day_key,
                            'temp': main_temp,
                            'temp_min': temp_min,
                            'temp_max': temp_max,
                            'pressure': pressure_matches[0] if pressure_matches else None,
                            'wind': wind_matches[0] if wind_matches else None,
                            'rain': rain_matches[0] if rain_matches else None,
                            'text': text[:200]
                        })
        
        found_days = found_days[:1]
        
        print(f"   Tìm thấy {len(found_days)} ngày dự báo (chỉ lấy hôm nay)")
        
        for idx, day_data in enumerate(found_days):
            forecast_date = today
            
            raw_text = day_data.get('text', '')
            raw_text = raw_text.replace('\n', ' ').replace('\r', ' ').strip()
            raw_text = ' '.join(raw_text.split())
            raw_text = raw_text[:100]
            
            record = {
                'city': CITY_MAPPING.get(city_key, city_key),
                'city_key': city_key,
                'date': forecast_date.strftime('%Y-%m-%d'),
                'source': 'thoitiet360',
                'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Temp': parse_temperature(day_data['temp']) if day_data['temp'] else None,
                'Temp_min': parse_temperature(day_data.get('temp_min')) if day_data.get('temp_min') else None,
                'Temp_max': parse_temperature(day_data.get('temp_max')) if day_data.get('temp_max') else None,
                'Pressure': parse_pressure(day_data['pressure']) if day_data['pressure'] else None,
                'Wind': parse_wind(day_data['wind']) if day_data['wind'] else None,
                'Rain': parse_rain(day_data['rain']) if day_data['rain'] else None,
                'Cloud': None,
                'raw_text': raw_text
            }
            
            forecast_data.append(record)
            temp_info = f"Temp={record['Temp']}°C"
            if record['Temp_min'] and record['Temp_max']:
                temp_info += f" (Thấp/Cao: {record['Temp_min']}°C/{record['Temp_max']}°C)"
            print(f"   ✓ Ngày {forecast_date.strftime('%Y-%m-%d')}: {temp_info}, Pressure={record['Pressure']}hPa, Wind={record['Wind']}km/h, Rain={record['Rain']}mm")
        
        if not forecast_data:
            print("   ⚠️  Không parse được dữ liệu bằng cách thông thường, thử cách khác...")
            print(f"   HTML sample (first 1000 chars): {response.text[:1000]}")
        
        return forecast_data
        
    except Exception as e:
        print(f"❌ Lỗi khi parse dữ liệu: {str(e)[:100]}")
        return []

def preprocess_thoitiet360_data(df):
    df = df.copy()
    
    city_mapping = {
        'Hà Nội': 'ha-noi',
        'Vinh': 'vinh',
        'Hồ Chí Minh': 'ho-chi-minh-city'
    }
    
    if 'city' in df.columns:
        df['city'] = df['city'].map(city_mapping).fillna(df['city'])
    
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' 00:00:00')
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    numeric_cols = ['Temp', 'Temp_min', 'Temp_max', 'Pressure', 'Wind', 'Rain', 'Cloud', 'Gust']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = None
    
    columns_to_keep = ['city', 'date', 'datetime', 'Temp', 'Temp_min', 'Temp_max', 'Pressure', 'Wind', 'Rain', 'Cloud', 'Gust']
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    columns_to_drop = ['raw_text', 'city_key', 'source', 'crawled_at', 'Time']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    if 'city' in df.columns and 'date' in df.columns:
        df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    return df

def save_to_csv(data, filename='thoitiet360_data.csv'):
    try:
        if not data:
            print("⚠️  Không có dữ liệu để lưu")
            return
        
        df_old = pd.DataFrame()
        import os
        if os.path.exists(filename):
            try:
                df_old = pd.read_csv(filename, encoding='utf-8-sig')
                print(f"📖 Đã đọc {len(df_old)} records cũ từ {filename}")
            except Exception as e:
                print(f"⚠️  Không thể đọc file cũ: {str(e)}")
                df_old = pd.DataFrame()
        
        df_new = pd.DataFrame(data)
        if 'raw_text' in df_new.columns:
            df_new['raw_text'] = df_new['raw_text'].astype(str).str.replace('\n', ' ').str.replace('\r', ' ').str.strip()
            df_new['raw_text'] = df_new['raw_text'].str[:100]
        
        if not df_old.empty:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['city', 'date'], keep='last')
            df_combined = df_combined.sort_values(['city', 'date']).reset_index(drop=True)
            df_final = df_combined
        else:
            df_final = df_new
        
        df_final.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu {len(df_final)} records vào {filename} (trong đó {len(df_new)} records mới)")
    except Exception as e:
        print(f"❌ Lỗi khi lưu CSV: {str(e)}")

def save_to_database(df):
    try:
        from database import init_database, get_db_connection
        import os
        
        init_database()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        inserted = 0
        updated = 0
        
        for _, row in df.iterrows():
            cursor.execute('''
                SELECT id FROM thoitiet360_data 
                WHERE city = ? AND date = ?
            ''', (row['city'], row['date']))
            
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE thoitiet360_data 
                    SET datetime = ?, Temp = ?, Temp_min = ?, Temp_max = ?, Pressure = ?, Wind = ?, Rain = ?, Cloud = ?, Gust = ?
                    WHERE city = ? AND date = ?
                ''', (
                    row.get('datetime'),
                    row.get('Temp'),
                    row.get('Temp_min'),
                    row.get('Temp_max'),
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
                cursor.execute('''
                    INSERT INTO thoitiet360_data 
                    (city, date, datetime, Temp, Temp_min, Temp_max, Pressure, Wind, Rain, Cloud, Gust)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['city'],
                    row['date'],
                    row.get('datetime'),
                    row.get('Temp'),
                    row.get('Temp_min'),
                    row.get('Temp_max'),
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
        
        if idx < len(cities):
            time.sleep(3)
    
    print(f"\n{'='*70}")
    print(f"TỔNG KẾT: Crawl được {len(all_data)} records")
    print(f"{'='*70}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        df_processed = preprocess_thoitiet360_data(df)
        
        save_to_csv(all_data, 'thoitiet360_data.csv')
        
        print("\n💾 Đang lưu vào database...")
        save_to_database(df_processed)
        
        print("\n📊 Tóm tắt dữ liệu:")
        if not df.empty:
            print(df.groupby('city').size())
            print("\nMẫu dữ liệu đã lưu:")
            print(df_processed[['city', 'date', 'Temp', 'Pressure', 'Wind', 'Rain']].head(10).to_string())
    else:
        print("⚠️  Không crawl được dữ liệu nào!")

if __name__ == '__main__':
    main()

