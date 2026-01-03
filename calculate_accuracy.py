"""
Script để tính toán độ chính xác và lưu vào database
Chạy script này để tính toán và lưu kết quả accuracy vào database
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sys
import io
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("="*70)
print("CALCULATING ACCURACY METRICS")
print("="*70)

def load_models():
    """Load models từ file .pkl"""
    model_file = 'weather_models_improved.pkl'
    if not os.path.exists(model_file):
        model_file = 'weather_models.pkl'
        print(f"⚠️  Không tìm thấy weather_models_improved.pkl, dùng {model_file}")
    
    if not os.path.exists(model_file):
        print(f"❌ File {model_file} không tồn tại!")
        print("   Vui lòng chạy: python train_improved_models.py trước")
        return None, None
    
    try:
        print(f"Loading models from {model_file}...")
        with open(model_file, 'rb') as f:
            models_data = pickle.load(f)
        
        models = models_data['models']
        feature_cols_dict = models_data['feature_cols']
        
        print(f"✅ Đã load {len(models)} models thành công!")
        return models, feature_cols_dict
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return None, None

def load_data():
    """Load data from SQLite database (fallback to CSV if database doesn't exist)"""
    try:
        from database import load_data_from_db, init_database
        import os
        
        if not os.path.exists('weather.db'):
            print("⚠️  Database not found. Falling back to CSV...")
            return load_data_from_csv()
        
        df = load_data_from_db()
        if len(df) == 0:
            print("⚠️  Database is empty. Falling back to CSV...")
            return load_data_from_csv()
        
        return df
    except Exception as e:
        print(f"⚠️  Error loading from database: {e}. Falling back to CSV...")
        return load_data_from_csv()

def load_data_from_csv():
    """Load data from CSV (fallback implementation)"""
    df = pd.read_csv('weather_all_cities.csv', encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    df['Temp'] = df['Temp'].str.replace(' °c', '').str.replace('°c', '').astype(float)
    df['Rain'] = df['Rain'].str.replace('mm', '').astype(float)
    df['Cloud'] = df['Cloud'].str.replace('%', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' mb', '').astype(float)
    df['Wind'] = df['Wind'].str.replace(' km/h', '').astype(float)
    df['Gust'] = df['Gust'].str.replace(' km/h', '').astype(float)
    
    dir_mapping = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
    df['Dir'] = df['Dir'].map(dir_mapping).fillna(0).astype(int)
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
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
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def create_features_for_prediction(df, target_col='Temp'):
    """Tạo advanced features cho dữ liệu cần predict"""
    df = df.copy()
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_data = df[city_mask].copy().sort_values('datetime').reset_index(drop=True)
        
        for lag in [12, 24]:
            df.loc[city_mask, f'{target_col}_lag_{lag}'] = city_data[target_col].shift(lag).values
        
        for window in [12, 24]:
            df.loc[city_mask, f'{target_col}_rolling_mean_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'{target_col}_rolling_std_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[city_mask, f'{target_col}_rolling_max_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).max().values
            df.loc[city_mask, f'{target_col}_rolling_min_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).min().values
        
        for lag in [1, 3, 6, 12]:
            df.loc[city_mask, f'Pressure_lag_{lag}'] = city_data['Pressure'].shift(lag).values
            df.loc[city_mask, f'Wind_lag_{lag}'] = city_data['Wind'].shift(lag).values
            df.loc[city_mask, f'Cloud_lag_{lag}'] = city_data['Cloud'].shift(lag).values
        
        for window in [3, 6, 12, 24]:
            df.loc[city_mask, f'Pressure_rolling_mean_{window}'] = city_data['Pressure'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Wind_rolling_mean_{window}'] = city_data['Wind'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Cloud_rolling_mean_{window}'] = city_data['Cloud'].shift(1).rolling(window=window, min_periods=1).mean().values
        
        city_data['hour'] = city_data['datetime'].dt.hour
        df.loc[city_mask, f'{target_col}_same_hour_1d_ago'] = city_data.groupby('hour')[target_col].shift(8).values
        df.loc[city_mask, f'{target_col}_same_hour_avg_7d'] = city_data.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
    
    df['hour_month_interaction'] = df['hour'] * df['month']
    df['pressure_wind_interaction'] = df['Pressure'] * df['Wind'] / 100
    df['cloud_hour_interaction'] = df['Cloud'] * df['hour'] / 100
    
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    return df

def calculate_accuracy():
    print("\n[1] Loading models...")
    models, feature_cols_dict = load_models()
    if models is None:
        return
    
    print("\n[2] Loading data...")
    df = load_data()
    
    test_start = '2023-01-01'
    test_end = '2025-12-15'
    test_df = df[
        (df['datetime'] >= test_start) & 
        (df['datetime'] <= test_end)
    ].copy()
    
    cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    result = {}
    
    overall_total_days = 0
    overall_ok_count = 0
    overall_bad_count = 0
    overall_total_min_error = 0
    overall_total_max_error = 0
    
    for city in cities:
        print(f"\n[3] Processing {city}...")
        city_df = test_df[test_df['city'] == city].copy()
        
        if len(city_df) == 0:
            continue
        
        city_df = city_df.sort_values('datetime').reset_index(drop=True)
        dates = sorted(city_df['datetime'].dt.date.unique())
        
        total_days = 0
        ok_count = 0
        bad_count = 0
        total_min_error = 0
        total_max_error = 0
        
        for idx, date in enumerate(dates):
            if (idx + 1) % 100 == 0:
                print(f"  Processing day {idx + 1}/{len(dates)}: {date}")
            
            day_data = city_df[city_df['datetime'].dt.date == date].copy()
            
            if len(day_data) == 0:
                continue
            
            day_data = day_data.sort_values('datetime')
            
            actual_min = float(day_data['Temp'].min())
            actual_max = float(day_data['Temp'].max())
            
            try:
                start_date_for_features = pd.to_datetime(date) - pd.Timedelta(days=2)
                data_for_features = city_df[
                    (city_df['datetime'] >= start_date_for_features) & 
                    (city_df['datetime'] <= pd.to_datetime(date) + pd.Timedelta(days=1))
                ].copy().sort_values('datetime')
                
                if len(data_for_features) == 0:
                    continue
                
                data_for_features = create_features_for_prediction(data_for_features, 'Temp')
                
                day_data_features = data_for_features[
                    data_for_features['datetime'].dt.date == date
                ].copy().sort_values('datetime')
                
                if len(day_data_features) == 0:
                    continue
                
                # Chọn model phù hợp: model riêng cho HCM hoặc model chung
                if city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                    model_key = 'Temp_numeric_hcm'
                else:
                    model_key = 'Temp_numeric'
                
                if model_key not in models:
                    continue
                
                feature_cols = feature_cols_dict.get(model_key, [])
                
                if len(feature_cols) > 0:
                    feature_mapping = {}
                    for col in feature_cols:
                        if col.endswith('_numeric') and col.replace('_numeric', '') in day_data_features.columns:
                            feature_mapping[col] = col.replace('_numeric', '')
                        elif col in day_data_features.columns:
                            feature_mapping[col] = col
                    
                    X = pd.DataFrame()
                    for model_col, data_col in feature_mapping.items():
                        if data_col in day_data_features.columns:
                            X[model_col] = day_data_features[data_col]
                        else:
                            X[model_col] = 0
                    
                    missing_features = [f for f in feature_cols if f not in X.columns]
                    if missing_features:
                        for f in missing_features:
                            X[f] = 0
                    
                    X = X[feature_cols].fillna(0)
                    predictions = models[model_key].predict(X)
                    
                    predicted_min = float(np.min(predictions))
                    predicted_max = float(np.max(predictions))
                    
                    min_error = abs(actual_min - predicted_min)
                    max_error = abs(actual_max - predicted_max)
                    
                    total_days += 1
                    total_min_error += min_error
                    total_max_error += max_error
                    
                    if min_error <= 2.0 and max_error <= 2.0:
                        ok_count += 1
                    else:
                        bad_count += 1
                        
            except Exception as e:
                continue
        
        if total_days > 0:
            avg_min_error = total_min_error / total_days
            avg_max_error = total_max_error / total_days
            ok_rate = (ok_count / total_days) * 100
            
            result[city] = {
                'total_days': int(total_days),
                'ok_count': int(ok_count),
                'bad_count': int(bad_count),
                'ok_rate': float(ok_rate),
                'avg_min_error': float(avg_min_error),
                'avg_max_error': float(avg_max_error)
            }
            
            print(f"  ✅ {city}: {ok_count}/{total_days} OK ({ok_rate:.1f}%)")
            
            overall_total_days += total_days
            overall_ok_count += ok_count
            overall_bad_count += bad_count
            overall_total_min_error += total_min_error
            overall_total_max_error += total_max_error
    
    if overall_total_days > 0:
        overall_ok_rate = (overall_ok_count / overall_total_days) * 100
        overall_avg_min_error = overall_total_min_error / overall_total_days
        overall_avg_max_error = overall_total_max_error / overall_total_days
        
        result['overall'] = {
            'total_days': int(overall_total_days),
            'ok_count': int(overall_ok_count),
            'bad_count': int(overall_bad_count),
            'ok_rate': float(overall_ok_rate),
            'avg_min_error': float(overall_avg_min_error),
            'avg_max_error': float(overall_avg_max_error),
            'start_date': test_start,
            'end_date': test_end,
            'cities': ['Vinh', 'Hà Nội', 'Hồ Chí Minh'],
            'calculated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    print("\n[4] Saving results to database...")
    
    try:
        from database import save_accuracy_results
        save_accuracy_results(result)
        print("✅ Đã lưu kết quả vào database")
    except Exception as e:
        print(f"❌ Lỗi khi lưu vào database: {e}")
        import traceback
        traceback.print_exc()
    print(f"\n📊 TỔNG HỢP:")
    print(f"   Tổng số ngày: {overall_total_days:,}")
    print(f"   OK: {overall_ok_count:,} ({overall_ok_rate:.1f}%)")
    print(f"   TỆ: {overall_bad_count:,} ({100 - overall_ok_rate:.1f}%)")
    print(f"   Sai số TB Min: {overall_avg_min_error:.2f}°C")
    print(f"   Sai số TB Max: {overall_avg_max_error:.2f}°C")
    print("\n" + "="*70)

if __name__ == '__main__':
    calculate_accuracy()

