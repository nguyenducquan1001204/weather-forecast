"""
Database models and connection for SQLite
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from contextlib import contextmanager

DB_PATH = 'weather.db'

def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    """Context manager for database connection"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_database():
    """Initialize database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create cold_air_settings table
    # level: 0=off, 1=nhẹ, 2=trung bình, 3=mạnh
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cold_air_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migrate from old is_active column to level column if needed
    try:
        cursor.execute('PRAGMA table_info(cold_air_settings)')
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'is_active' in columns and 'level' not in columns:
            # Migrate: add level column and convert is_active to level
            cursor.execute('ALTER TABLE cold_air_settings ADD COLUMN level INTEGER DEFAULT 0')
            cursor.execute('UPDATE cold_air_settings SET level = is_active WHERE level = 0')
        elif 'is_active' in columns and 'level' in columns:
            # Both columns exist, migrate data
            cursor.execute('UPDATE cold_air_settings SET level = is_active WHERE level = 0 AND is_active IS NOT NULL')
    except:
        pass
    
    # Insert default setting if not exists
    cursor.execute('SELECT COUNT(*) FROM cold_air_settings')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO cold_air_settings (level) VALUES (0)
        ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            date TEXT NOT NULL,
            Time TEXT NOT NULL,
            datetime TEXT NOT NULL,
            Temp REAL,
            Rain REAL,
            Cloud REAL,
            Pressure REAL,
            Wind REAL,
            Gust REAL,
            Dir INTEGER,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            hour INTEGER,
            day_of_year INTEGER,
            day_of_week INTEGER,
            is_weekend INTEGER,
            season INTEGER,
            hour_sin REAL,
            hour_cos REAL,
            month_sin REAL,
            month_cos REAL,
            day_of_year_sin REAL,
            day_of_year_cos REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_city_year_month_day_hour ON weather_data(city, year, month, day, hour)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_city_year ON weather_data(city, year)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_city_year_month ON weather_data(city, year, month)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_city_year_month_day ON weather_data(city, year, month, day)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_datetime ON weather_data(datetime)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON weather_data(date)')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accuracy_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            total_days INTEGER NOT NULL,
            ok_count INTEGER NOT NULL,
            bad_count INTEGER NOT NULL,
            ok_rate REAL NOT NULL,
            avg_min_error REAL NOT NULL,
            avg_max_error REAL NOT NULL,
            start_date TEXT,
            end_date TEXT,
            calculated_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_accuracy_city ON accuracy_results(city)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_accuracy_calculated_at ON accuracy_results(calculated_at)')
    
    # Create table for thoitiet360 comparison data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thoitiet360_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            date TEXT NOT NULL,
            datetime TEXT NOT NULL,
            Temp REAL,
            Temp_min REAL,
            Temp_max REAL,
            Pressure REAL,
            Wind REAL,
            Rain REAL,
            Cloud REAL,
            Gust REAL,
            crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, date)
        )
    ''')
    
    # Thêm cột Temp_min và Temp_max nếu chưa có (migration)
    try:
        cursor.execute('ALTER TABLE thoitiet360_data ADD COLUMN Temp_min REAL')
    except:
        pass  # Cột đã tồn tại
    
    try:
        cursor.execute('ALTER TABLE thoitiet360_data ADD COLUMN Temp_max REAL')
    except:
        pass  # Cột đã tồn tại
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoitiet360_city_date ON thoitiet360_data(city, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_thoitiet360_datetime ON thoitiet360_data(datetime)')
    
    # Migration: Thêm Temp_min và Temp_max vào system_forecasts nếu chưa có
    try:
        cursor.execute('ALTER TABLE system_forecasts ADD COLUMN Temp_min REAL')
    except:
        pass  # Cột đã tồn tại
    
    try:
        cursor.execute('ALTER TABLE system_forecasts ADD COLUMN Temp_max REAL')
    except:
        pass  # Cột đã tồn tại
    
    # Create table for system forecasts (để so sánh với thoitiet360)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            date TEXT NOT NULL,
            Temp REAL,
            Temp_min REAL,
            Temp_max REAL,
            Pressure REAL,
            Wind REAL,
            Rain REAL,
            Cloud REAL,
            Gust REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, date)
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_forecasts_city_date ON system_forecasts(city, date)')
    
    conn.commit()
    conn.close()
    print("[OK] Database schema initialized successfully!")

def load_data_from_db():
    """Load data from database and return as DataFrame (similar to original load_data function)"""
    conn = get_db_connection()
    
    query = '''
        SELECT 
            city, date, Time, datetime,
            Temp, Rain, Cloud, Pressure, Wind, Gust, Dir,
            year, month, day, hour,
            day_of_year, day_of_week, is_weekend, season,
            hour_sin, hour_cos, month_sin, month_cos,
            day_of_year_sin, day_of_year_cos
        FROM weather_data
        ORDER BY city, datetime
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df

def get_data_count():
    """Get total number of records in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM weather_data')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def clean_weather_data(df):
    """
    Clean and preprocess weather data (similar to load_data function)
    Returns cleaned DataFrame ready for database insertion
    """
    df = df.copy()
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # Clean data columns - remove units
    if 'Temp' in df.columns:
        df['Temp'] = df['Temp'].astype(str).str.replace(' °c', '').str.replace('°c', '').str.replace('°C', '').str.strip()
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
    
    if 'Rain' in df.columns:
        df['Rain'] = df['Rain'].astype(str).str.replace('mm', '').str.strip()
        df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce')
    
    if 'Cloud' in df.columns:
        df['Cloud'] = df['Cloud'].astype(str).str.replace('%', '').str.strip()
        df['Cloud'] = pd.to_numeric(df['Cloud'], errors='coerce')
    
    if 'Pressure' in df.columns:
        df['Pressure'] = df['Pressure'].astype(str).str.replace(' mb', '').str.replace('mb', '').str.strip()
        df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
    
    if 'Wind' in df.columns:
        df['Wind'] = df['Wind'].astype(str).str.replace(' km/h', '').str.replace('km/h', '').str.strip()
        df['Wind'] = pd.to_numeric(df['Wind'], errors='coerce')
    
    if 'Gust' in df.columns:
        df['Gust'] = df['Gust'].astype(str).str.replace(' km/h', '').str.replace('km/h', '').str.strip()
        df['Gust'] = pd.to_numeric(df['Gust'], errors='coerce')
    
    # Encode Dir column
    if 'Dir' in df.columns:
        dir_mapping = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
        df['Dir'] = df['Dir'].map(dir_mapping).fillna(0).astype(int)
    
    # Extract datetime components
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Season calculation
    def get_season(month):
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3
    
    df['season'] = df['month'].apply(get_season)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Convert datetime to string for database storage
    df['datetime'] = df['datetime'].astype(str)
    
    return df

def insert_data_to_db(df, batch_size=1000, skip_duplicates=True):
    """
    Insert cleaned weather data into database
    Args:
        df: DataFrame with cleaned data
        batch_size: Number of records to insert per batch
        skip_duplicates: If True, skip records that already exist (based on city, date, Time)
    Returns:
        Number of records inserted
    """
    if len(df) == 0:
        return 0
    
    # Ensure database is initialized
    init_database()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get existing records if skip_duplicates is True
    existing_records = set()
    if skip_duplicates:
        cursor.execute('SELECT city, date, Time FROM weather_data')
        existing_records = set(cursor.fetchall())
    
    # Prepare data for insertion
    columns = [
        'city', 'date', 'Time', 'datetime',
        'Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir',
        'year', 'month', 'day', 'hour',
        'day_of_year', 'day_of_week', 'is_weekend', 'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos'
    ]
    
    # Filter out columns that don't exist
    available_columns = [col for col in columns if col in df.columns]
    
    # Prepare values for insertion
    values_to_insert = []
    for _, row in df.iterrows():
        # Check for duplicates
        if skip_duplicates:
            record_key = (str(row.get('city', '')), str(row.get('date', '')), str(row.get('Time', '')))
            if record_key in existing_records:
                continue
        
        values = [row.get(col) for col in available_columns]
        values_to_insert.append(values)
    
    if len(values_to_insert) == 0:
        conn.close()
        return 0
    
    # Insert in batches
    placeholders = ','.join(['?' for _ in available_columns])
    query = f'''
        INSERT INTO weather_data ({','.join(available_columns)})
        VALUES ({placeholders})
    '''
    
    inserted = 0
    for i in range(0, len(values_to_insert), batch_size):
        batch = values_to_insert[i:i+batch_size]
        cursor.executemany(query, batch)
        conn.commit()
        inserted += len(batch)
    
    conn.close()
    return inserted

def save_accuracy_results(result_dict):
    """
    Save accuracy results to database
    Args:
        result_dict: Dictionary with accuracy results (from calculate_accuracy.py)
    Returns:
        Number of records inserted
    """
    # Ensure database schema is initialized
    init_database()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    inserted = 0
    
    for key, data in result_dict.items():
        if key == 'overall':
            cursor.execute('''
                INSERT INTO accuracy_results 
                (city, total_days, ok_count, bad_count, ok_rate, avg_min_error, avg_max_error, 
                 start_date, end_date, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                None,
                data.get('total_days', 0),
                data.get('ok_count', 0),
                data.get('bad_count', 0),
                data.get('ok_rate', 0.0),
                data.get('avg_min_error', 0.0),
                data.get('avg_max_error', 0.0),
                data.get('start_date'),
                data.get('end_date'),
                data.get('calculated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ))
        else:
            cursor.execute('''
                INSERT INTO accuracy_results 
                (city, total_days, ok_count, bad_count, ok_rate, avg_min_error, avg_max_error, 
                 start_date, end_date, calculated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                key,
                data.get('total_days', 0),
                data.get('ok_count', 0),
                data.get('bad_count', 0),
                data.get('ok_rate', 0.0),
                data.get('avg_min_error', 0.0),
                data.get('avg_max_error', 0.0),
                None,
                None,
                result_dict.get('overall', {}).get('calculated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            ))
        inserted += 1
    
    conn.commit()
    conn.close()
    return inserted

def get_latest_accuracy_results():
    """
    Get the latest accuracy results from database
    Returns:
        Dictionary with accuracy results (similar to JSON structure)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT MAX(calculated_at) as latest_date
        FROM accuracy_results
    ''')
    latest_date = cursor.fetchone()[0]
    
    if not latest_date:
        conn.close()
        return None
    
    cursor.execute('''
        SELECT * FROM accuracy_results
        WHERE calculated_at = ?
        ORDER BY 
            CASE WHEN city IS NULL THEN 1 ELSE 0 END,
            city
    ''', (latest_date,))
    
    rows = cursor.fetchall()
    conn.close()
    
    result = {}
    for row in rows:
        if row['city'] is None:
            result['overall'] = {
                'total_days': row['total_days'],
                'ok_count': row['ok_count'],
                'bad_count': row['bad_count'],
                'ok_rate': row['ok_rate'],
                'avg_min_error': row['avg_min_error'],
                'avg_max_error': row['avg_max_error'],
                'start_date': row['start_date'],
                'end_date': row['end_date'],
                'calculated_at': row['calculated_at']
            }
        else:
            result[row['city']] = {
                'total_days': row['total_days'],
                'ok_count': row['ok_count'],
                'bad_count': row['bad_count'],
                'ok_rate': row['ok_rate'],
                'avg_min_error': row['avg_min_error'],
                'avg_max_error': row['avg_max_error']
            }
    
    return result

def get_thoitiet360_data(city, date=None):
    """
    Lấy dữ liệu từ thoitiet360.edu.vn để so sánh
    Args:
        city: Tên thành phố (vinh, ha-noi, ho-chi-minh-city)
        date: Ngày cần lấy (YYYY-MM-DD), nếu None thì lấy ngày mới nhất
    Returns:
        Dict chứa dữ liệu hoặc None nếu không tìm thấy
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if date:
        cursor.execute('''
            SELECT * FROM thoitiet360_data
            WHERE city = ? AND date = ?
            ORDER BY crawled_at DESC
            LIMIT 1
        ''', (city, date))
    else:
        cursor.execute('''
            SELECT * FROM thoitiet360_data
            WHERE city = ?
            ORDER BY date DESC, crawled_at DESC
            LIMIT 1
        ''', (city,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        # sqlite3.Row không có method get(), cần dùng try-except hoặc truy cập trực tiếp
        def safe_get(row, key, default=None):
            try:
                return row[key] if row[key] is not None else default
            except (KeyError, IndexError):
                return default
        
        return {
            'city': row['city'],
            'date': row['date'],
            'Temp': row['Temp'],
            'Temp_min': safe_get(row, 'Temp_min'),
            'Temp_max': safe_get(row, 'Temp_max'),
            'Pressure': row['Pressure'],
            'Wind': row['Wind'],
            'Rain': row['Rain'],
            'Cloud': row['Cloud'],
            'Gust': row['Gust'],
            'crawled_at': row['crawled_at']
        }
    return None

