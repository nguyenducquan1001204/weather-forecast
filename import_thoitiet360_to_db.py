
import pandas as pd
from database import init_database, get_db_connection
import os
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def import_csv_to_database():
    """Import dữ liệu từ CSV vào database"""
    csv_file = 'thoitiet360_data.csv'
    
    if not os.path.exists(csv_file):
        print(f"⚠️  Không tìm thấy file {csv_file}")
        return 0
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        
        if df.empty:
            print("⚠️  File CSV trống")
            return 0
        
        init_database()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        inserted = 0
        updated = 0
        
        for _, row in df.iterrows():
            city = row.get('city_key') or row.get('city', '')
            date = row.get('date', '')
            
            if not city or not date:
                continue
            
            city_mapping = {
                'Hà Nội': 'ha-noi',
                'Vinh': 'vinh',
                'Hồ Chí Minh': 'ho-chi-minh-city'
            }
            if city in city_mapping:
                city = city_mapping[city]
            
            cursor.execute('''
                SELECT id FROM thoitiet360_data 
                WHERE city = ? AND date = ?
            ''', (city, date))
            
            existing = cursor.fetchone()
            
            datetime_val = row.get('datetime')
            if pd.isna(datetime_val) or not datetime_val:
                datetime_val = f"{date} 00:00:00"
            
            if existing:
                cursor.execute('''
                    UPDATE thoitiet360_data 
                    SET datetime = ?, Temp = ?, Pressure = ?, Wind = ?, Rain = ?, Cloud = ?, Gust = ?
                    WHERE city = ? AND date = ?
                ''', (
                    datetime_val,
                    row.get('Temp') if pd.notna(row.get('Temp')) else None,
                    row.get('Pressure') if pd.notna(row.get('Pressure')) else None,
                    row.get('Wind') if pd.notna(row.get('Wind')) else None,
                    row.get('Rain') if pd.notna(row.get('Rain')) else None,
                    row.get('Cloud') if pd.notna(row.get('Cloud')) else None,
                    row.get('Gust') if pd.notna(row.get('Gust')) else None,
                    city,
                    date
                ))
                updated += 1
            else:
                cursor.execute('''
                    INSERT INTO thoitiet360_data 
                    (city, date, datetime, Temp, Pressure, Wind, Rain, Cloud, Gust)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    city,
                    date,
                    datetime_val,
                    row.get('Temp') if pd.notna(row.get('Temp')) else None,
                    row.get('Pressure') if pd.notna(row.get('Pressure')) else None,
                    row.get('Wind') if pd.notna(row.get('Wind')) else None,
                    row.get('Rain') if pd.notna(row.get('Rain')) else None,
                    row.get('Cloud') if pd.notna(row.get('Cloud')) else None,
                    row.get('Gust') if pd.notna(row.get('Gust')) else None
                ))
                inserted += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Đã import vào database: {inserted} records mới, {updated} records cập nhật")
        return inserted + updated
        
    except Exception as e:
        print(f"❌ Lỗi khi import vào database: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == '__main__':
    print("="*70)
    print("IMPORT DỮ LIỆU THOITIET360 TỪ CSV VÀO DATABASE")
    print("="*70)
    import_csv_to_database()

