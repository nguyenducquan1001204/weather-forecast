from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class WeatherCrawler:
    def __init__(self, city="vinh", country="vn"):
        self.base_url = "https://www.worldweatheronline.com"
        self.city = city
        self.country = country
        self.driver = None
        self._init_driver()
    
    def _init_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        if sys.platform == 'linux':
            chrome_options.binary_location = os.environ.get('CHROME_BIN', '/usr/bin/chromium-browser')
            chromedriver_path = os.environ.get('CHROMEDRIVER_PATH', '/usr/bin/chromedriver')
            if os.path.exists(chromedriver_path):
                service = Service(chromedriver_path)
            else:
                service = Service()
        else:
            service = None
        
        try:
            if service:
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi tạo Chrome driver: {str(e)}")
            print("Vui lòng cài đặt ChromeDriver hoặc sử dụng trình duyệt khác")
            raise
    
    def __del__(self):
        if self.driver:
            self.driver.quit()
    
    def crawl_weather_by_date(self, date: str, retry_count: int = 2) -> Optional[Dict]:
        for attempt in range(retry_count + 1):
            try:
                if '-' in date:
                    parts = date.split('-')
                    if len(parts[0]) == 4:
                        day, month, year = parts[2], parts[1], parts[0]
                    else:
                        day, month, year = parts[0], parts[1], parts[2]
                else:
                    return None
                
                url = f"{self.base_url}/{self.city}-weather-history/{self.country}.aspx"
                
                date_str = f"{year}-{month}-{day}"
                if attempt > 0:
                    print(f"  -> Thử lại lần {attempt + 1}...")
                else:
                    print(f"Đang crawl dữ liệu cho ngày {day}/{month}/{year}...")
                print(f"URL: {url}")
                
                if not self.driver:
                    self._init_driver()
                
                self.driver.get(url)
                time.sleep(3)
                
                date_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//input[contains(@id, 'txtPastDate')]"))
                )
                
                self.driver.execute_script(f"arguments[0].value = '{date_str}';", date_input)
                time.sleep(0.5)
                
                get_weather_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@value='Get Weather']"))
                )
                get_weather_button.click()
                
                try:
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "days-details-table"))
                    )
                    time.sleep(4)
                except:
                    time.sleep(5)
                
                from io import StringIO
                dfs = pd.read_html(StringIO(self.driver.page_source))
                
                weather_data = None
                for df in dfs:
                    if 'Time' in df.columns and 'Pressure' in df.columns:
                        df = df[df['Time'] != 'Time'].copy()
                        if len(df) > 0:
                            rows_data = df.to_dict('records')
                            weather_data = {
                                'hourly_data': rows_data,
                                'summary': {}
                            }
                            print(f"  -> OK: Lấy được {len(df)} dòng bằng pd.read_html.")
                            break
                
                if not weather_data:
                    print(f"  -> Thử parse bằng BeautifulSoup...")
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    weather_data = self._parse_weather_table(soup, date)
                    if weather_data and weather_data.get('hourly_data'):
                        print(f"  -> OK: Lấy được {len(weather_data['hourly_data'])} dòng bằng BeautifulSoup.")
                
                if weather_data and weather_data.get('hourly_data') and len(weather_data['hourly_data']) > 0:
                    weather_data['date'] = f"{year}-{month}-{day}"
                    weather_data['city'] = self.city
                    weather_data['country'] = self.country
                    return weather_data
                elif attempt < retry_count:
                    print(f"  -> Không có dữ liệu, thử lại...")
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi crawl ngày {date_str if 'date_str' in locals() else date}: {str(e)}")
                if attempt < retry_count:
                    print(f"  -> Thử lại...")
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def _extract_cell_text(self, cell, header: str) -> str:
        text = cell.get_text(strip=True)
        if text:
            return text
        
        img = cell.find('img')
        if img:
            alt = img.get('alt', '')
            title = img.get('title', '')
            src = img.get('src', '')
            if alt:
                return alt
            if title:
                return title
            if 'north' in src.lower() or 'n' in src.lower():
                return 'N'
            if 'south' in src.lower() or 's' in src.lower():
                return 'S'
            if 'east' in src.lower() or 'e' in src.lower():
                return 'E'
            if 'west' in src.lower() or 'w' in src.lower():
                return 'W'
            if 'northeast' in src.lower() or 'ne' in src.lower():
                return 'NE'
            if 'northwest' in src.lower() or 'nw' in src.lower():
                return 'NW'
            if 'southeast' in src.lower() or 'se' in src.lower():
                return 'SE'
            if 'southwest' in src.lower() or 'sw' in src.lower():
                return 'SW'
        
        svg = cell.find('svg')
        if svg:
            import re
            svg_style = svg.get('style', '')
            if svg_style and ('transform' in svg_style.lower() or 'rotate' in svg_style.lower()):
                rotate_match = re.search(r'rotate\(([^)]+)\)', svg_style)
                if rotate_match:
                    angle = rotate_match.group(1).replace('deg', '').strip()
                    try:
                        angle_float = float(angle)
                        if 337.5 <= angle_float or angle_float < 22.5:
                            return 'N'
                        elif 22.5 <= angle_float < 67.5:
                            return 'NE'
                        elif 67.5 <= angle_float < 112.5:
                            return 'E'
                        elif 112.5 <= angle_float < 157.5:
                            return 'SE'
                        elif 157.5 <= angle_float < 202.5:
                            return 'S'
                        elif 202.5 <= angle_float < 247.5:
                            return 'SW'
                        elif 247.5 <= angle_float < 292.5:
                            return 'W'
                        elif 292.5 <= angle_float < 337.5:
                            return 'NW'
                    except:
                        pass
            title_tag = svg.find('title')
            if title_tag:
                return title_tag.get_text(strip=True)
            use_tag = svg.find('use')
            if use_tag:
                href = use_tag.get('href', '')
                xlink_href = use_tag.get('xlink:href', '')
                href_val = href or xlink_href
                if 'north' in href_val.lower() or 'n' in href_val.lower():
                    return 'N'
                if 'south' in href_val.lower() or 's' in href_val.lower():
                    return 'S'
                if 'east' in href_val.lower() or 'e' in href_val.lower():
                    return 'E'
                if 'west' in href_val.lower() or 'w' in href_val.lower():
                    return 'W'
        
        aria_label = cell.get('aria-label', '')
        if aria_label:
            return aria_label
        
        title_attr = cell.get('title', '')
        if title_attr:
            return title_attr
        
        data_attrs = [cell.get(attr) for attr in cell.attrs if attr.startswith('data-')]
        for data_val in data_attrs:
            if data_val and isinstance(data_val, str):
                return data_val
        
        if header.lower() == 'dir' or header.lower() == 'direction':
            import re
            
            all_elements = cell.find_all(['span', 'div', 'i', 'svg', 'img'])
            for elem in all_elements:
                elem_text = elem.get_text(strip=True)
                if elem_text:
                    return elem_text
                
                elem_style = elem.get('style', '')
                if elem_style and ('transform' in elem_style.lower() or 'rotate' in elem_style.lower()):
                    rotate_match = re.search(r'rotate\(([^)]+)\)', elem_style)
                    if rotate_match:
                        angle = rotate_match.group(1).replace('deg', '').strip()
                        try:
                            angle_float = float(angle)
                            if 337.5 <= angle_float or angle_float < 22.5:
                                return 'N'
                            elif 22.5 <= angle_float < 67.5:
                                return 'NE'
                            elif 67.5 <= angle_float < 112.5:
                                return 'E'
                            elif 112.5 <= angle_float < 157.5:
                                return 'SE'
                            elif 157.5 <= angle_float < 202.5:
                                return 'S'
                            elif 202.5 <= angle_float < 247.5:
                                return 'SW'
                            elif 247.5 <= angle_float < 292.5:
                                return 'W'
                            elif 292.5 <= angle_float < 337.5:
                                return 'NW'
                        except:
                            pass
                
                elem_class = elem.get('class', [])
                if elem_class:
                    class_str = ' '.join(elem_class).lower()
                    if 'n' in class_str and 'e' in class_str:
                        return 'NE'
                    if 'n' in class_str and 'w' in class_str:
                        return 'NW'
                    if 's' in class_str and 'e' in class_str:
                        return 'SE'
                    if 's' in class_str and 'w' in class_str:
                        return 'SW'
                    if 'n' in class_str:
                        return 'N'
                    if 's' in class_str:
                        return 'S'
                    if 'e' in class_str:
                        return 'E'
                    if 'w' in class_str:
                        return 'W'
            
            cell_style = cell.get('style', '')
            if cell_style and ('transform' in cell_style.lower() or 'rotate' in cell_style.lower()):
                rotate_match = re.search(r'rotate\(([^)]+)\)', cell_style)
                if rotate_match:
                    angle = rotate_match.group(1).replace('deg', '').strip()
                    try:
                        angle_float = float(angle)
                        if 337.5 <= angle_float or angle_float < 22.5:
                            return 'N'
                        elif 22.5 <= angle_float < 67.5:
                            return 'NE'
                        elif 67.5 <= angle_float < 112.5:
                            return 'E'
                        elif 112.5 <= angle_float < 157.5:
                            return 'SE'
                        elif 157.5 <= angle_float < 202.5:
                            return 'S'
                        elif 202.5 <= angle_float < 247.5:
                            return 'SW'
                        elif 247.5 <= angle_float < 292.5:
                            return 'W'
                        elif 292.5 <= angle_float < 337.5:
                            return 'NW'
                    except:
                        pass
        
        return ''
    
    def _parse_weather_table(self, soup: BeautifulSoup, date: str) -> Optional[Dict]:
        try:
            table = soup.find('table', class_='days-details-table')
            
            if not table:
                tables = soup.find_all('table')
                for t in tables:
                    text = t.get_text().lower()
                    if 'time' in text and ('temp' in text or 'weather' in text):
                        table = t
                        break
            
            if not table:
                print("Không tìm thấy bảng dữ liệu")
                page_title = soup.find('title')
                if page_title:
                    print(f"Tiêu đề trang: {page_title.get_text()}")
                return None
            
            headers = []
            header_row = table.find('tr', class_='days-details-row-header')
            if not header_row:
                header_row = table.find('tr')
            
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    header_text = th.get_text(strip=True)
                    if header_text:
                        headers.append(header_text)
            
            rows_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                if 'days-details-row-header' in row.get('class', []) or 'divider' in row.get('class', []):
                    continue
                
                cells = row.find_all(['td', 'th'])
                if len(cells) > 0:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            header = headers[i]
                            cell_text = self._extract_cell_text(cell, header)
                            row_data[header] = cell_text if cell_text else ''
                    
                    if any(row_data.values()):
                        rows_data.append(row_data)
            
            summary = {}
            page_text = soup.get_text()
            if 'min/max' in page_text.lower():
                summary['has_summary'] = True
            
            return {
                'hourly_data': rows_data,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Lỗi khi parse dữ liệu: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def crawl_multiple_dates(self, start_date: str, end_date: str, output_file: str = "weather_data.csv"):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_data = []
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            weather_data = self.crawl_weather_by_date(date_str)
            
            if weather_data:
                for hourly in weather_data.get('hourly_data', []):
                    record = {
                        'date': weather_data['date'],
                        'city': weather_data['city'],
                        'country': weather_data['country'],
                        **hourly
                    }
                    all_data.append(record)
            
            time.sleep(2)
            
            current_date += timedelta(days=1)
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nĐã lưu {len(all_data)} bản ghi vào {output_file}")
        else:
            print("Không có dữ liệu để lưu")
    
    def crawl_single_date_to_csv(self, date: str, output_file: str = None):
        if output_file is None:
            output_file = f"weather_{date.replace('-', '')}.csv"
        
        weather_data = self.crawl_weather_by_date(date)
        
        if weather_data and weather_data.get('hourly_data'):
            df = pd.DataFrame(weather_data['hourly_data'])
            df.insert(0, 'date', weather_data['date'])
            df.insert(1, 'city', weather_data['city'])
            df.insert(2, 'country', weather_data['country'])
            
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nĐã lưu dữ liệu vào {output_file}")
            return df
        else:
            print("Không có dữ liệu để lưu")
            return None


def main():
    cities = [
        "vinh",
        "ha-noi",
        "ho-chi-minh-city"
    ]
    
    try:
        import pytz
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        today_vn = datetime.now(vn_tz).date()
    except ImportError:
        utc_now = datetime.now(timezone.utc)
        vn_offset = timezone(timedelta(hours=7))
        today_vn = (utc_now.astimezone(vn_offset)).date()
    
    yesterday = today_vn - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    start_date = yesterday_str
    end_date = yesterday_str
    
    output_file = "weather_all_cities.csv"
    save_csv_backup = True
    
    print(f"=== CRAWL DỮ LIỆU THỜI TIẾT ===")
    print(f"Khoảng thời gian: {start_date} -> {end_date}")
    print(f"Database: weather.db")
    if save_csv_backup:
        print(f"CSV backup: {output_file}")
    print()
    
    from database import init_database, get_db_connection, clean_weather_data, insert_data_to_db
    init_database()
    
    existing_city_dates = set()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT city, date FROM weather_data')
        for row in cursor.fetchall():
            existing_city_dates.add((row[0], row[1]))
        conn.close()
        
        if len(existing_city_dates) > 0:
            existing_cities = set([c for c, _ in existing_city_dates])
            existing_dates = sorted(set([d for _, d in existing_city_dates]))
            latest_date = max(existing_dates) if existing_dates else None
            print(f"Đã tìm thấy dữ liệu trong database:")
            print(f"  - {len(existing_cities)} tỉnh/ thành phố: {', '.join(sorted(existing_cities))}")
            print(f"  - {len(existing_dates)} ngày (ngày mới nhất: {latest_date})")
            print()
    except Exception as e:
        print(f"Cảnh báo: Không thể đọc database: {str(e)}\n")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates_to_crawl = []
    current_date = start
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        dates_to_crawl.append(date_str)
        current_date += timedelta(days=1)
    
    city_dates_to_crawl = []
    for city in cities:
        for date_str in dates_to_crawl:
            if (city, date_str) not in existing_city_dates:
                city_dates_to_crawl.append((city, date_str))
    
    if not city_dates_to_crawl:
        print("Tất cả các ngày và thành phố đã được crawl. Không cần crawl thêm.")
        return
    
    cities_to_crawl = {}
    for city, date_str in city_dates_to_crawl:
        if city not in cities_to_crawl:
            cities_to_crawl[city] = []
        cities_to_crawl[city].append(date_str)
    
    print(f"Số cặp (thành phố, ngày) cần crawl: {len(city_dates_to_crawl)}")
    for city, dates in cities_to_crawl.items():
        print(f"  - {city}: {len(dates)} ngày ({', '.join(sorted(dates))})")
    print()
    
    all_data = []
    new_records_count = 0
    
    for i, (city, dates) in enumerate(cities_to_crawl.items(), 1):
        print(f"\n[{i}/{len(cities_to_crawl)}] Đang crawl dữ liệu cho {city}...")
        crawler = WeatherCrawler(city=city, country="vn")
        
        try:
            city_data_count = 0
            for date_str in sorted(dates):
                print(f"  -> Crawl ngày {date_str}...")
                weather_data = crawler.crawl_weather_by_date(date_str)
                
                if weather_data:
                    for hourly in weather_data.get('hourly_data', []):
                        record = {
                            'date': weather_data['date'],
                            'city': weather_data['city'],
                            'country': weather_data['country'],
                            **hourly
                        }
                        all_data.append(record)
                        city_data_count += 1
                        new_records_count += 1
                
                time.sleep(2)
            
            print(f"✓ Hoàn thành {city} - crawl được {city_data_count} bản ghi")
        except Exception as e:
            print(f"✗ Lỗi khi crawl {city}: {str(e)}")
        
        time.sleep(3)
    
    if new_records_count > 0:
        print(f"\n[Processing] Đang xử lý {new_records_count} bản ghi mới...")
        
        df_raw = pd.DataFrame(all_data)
        
        print("  -> Làm sạch dữ liệu...")
        df_cleaned = clean_weather_data(df_raw)
        
        print("  -> Lưu vào database...")
        inserted_count = insert_data_to_db(df_cleaned, batch_size=1000, skip_duplicates=True)
        
        print(f"\n=== HOÀN THÀNH ===")
        print(f"Đã crawl: {new_records_count} bản ghi")
        print(f"Đã lưu vào database: {inserted_count} bản ghi (đã bỏ qua trùng lặp)")
        
        if save_csv_backup:
            try:
                if os.path.exists(output_file):
                    df_existing = pd.read_csv(output_file, encoding='utf-8-sig')
                    df_csv = pd.concat([df_existing, df_raw], ignore_index=True)
                    df_csv = df_csv.drop_duplicates(subset=['city', 'date', 'Time'], keep='last')
                else:
                    df_csv = df_raw
                
                df_csv.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"Đã lưu CSV backup: {output_file} ({len(df_csv)} bản ghi)")
            except Exception as e:
                print(f"Cảnh báo: Không thể lưu CSV backup: {str(e)}")
        
        from database import get_data_count
        total_in_db = get_data_count()
        print(f"Tổng số bản ghi trong database: {total_in_db}")
    else:
        print(f"\n=== THÔNG BÁO ===")
        print("Không có dữ liệu mới để thêm.")
        from database import get_data_count
        total_in_db = get_data_count()
        if total_in_db > 0:
            print(f"Database hiện có {total_in_db} bản ghi.")


if __name__ == "__main__":
    main()

