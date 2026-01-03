from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import io
import pickle
import os
import json

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)

models = {}
feature_cols_dict = {}
target_vars = ['Temp_numeric', 'Rain_numeric', 'Cloud_numeric', 
               'Pressure_numeric', 'Wind_numeric', 'Gust_numeric']

def load_models():
    """Load models từ file .pkl"""
    global models, feature_cols_dict
    
    # Ưu tiên load final model (train với toàn bộ dữ liệu), sau đó improved, cuối cùng là model cũ
    model_file = 'weather_models_final.pkl'
    if not os.path.exists(model_file):
        model_file = 'weather_models_improved.pkl'
        print(f"⚠️  Không tìm thấy weather_models_final.pkl, dùng {model_file}")
        if not os.path.exists(model_file):
            model_file = 'weather_models.pkl'
            print(f"⚠️  Không tìm thấy weather_models_improved.pkl, dùng {model_file}")
    
    if not os.path.exists(model_file):
        print(f"❌ File {model_file} không tồn tại!")
        print("   Vui lòng chạy: python train_improved_models.py trước")
        return False
    
    try:
        print(f"Loading models from {model_file}...")
        with open(model_file, 'rb') as f:
            models_data = pickle.load(f)
        
        models = models_data['models']
        feature_cols_dict = models_data['feature_cols']
        
        print(f"✅ Đã load {len(models)} models thành công!")
        print(f"   Models: {list(models.keys())}")
        if model_file == 'weather_models_final.pkl':
            print(f"   Using FINAL models (trained on ALL data 2017-2026-01-02)")
            if 'Temp_numeric' in models:
                print(f"   - Temp model: R² ~97%")
            if 'Temp_numeric_hcm' in models:
                print(f"   - HCM Temp model: R² ~97.6%")
        elif 'Temp_numeric' in models:
            print(f"   Using improved temperature model (R² ~94%)")
            if 'Temp_numeric_hcm' in models:
                print(f"   Using HCM-specific temperature model (R² ~91.6%)")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return False

def load_data():
    """Load data from SQLite database (fallback to CSV if database doesn't exist)"""
    try:
        from database import load_data_from_db, init_database
        import os
        
        # Initialize database if it doesn't exist
        if not os.path.exists('weather.db'):
            print("⚠️  Database not found. Initializing...")
            init_database()
            print("⚠️  Database is empty. Please run migrate_csv_to_db.py first.")
            print("⚠️  Falling back to CSV for now...")
            return load_data_from_csv()
        
        # Try to load from database
        df = load_data_from_db()
        if len(df) == 0:
            print("⚠️  Database is empty. Please run migrate_csv_to_db.py first.")
            print("⚠️  Falling back to CSV for now...")
            return load_data_from_csv()
        
        return df
    except Exception as e:
        print(f"⚠️  Error loading from database: {e}")
        print("⚠️  Falling back to CSV...")
        return load_data_from_csv()

def load_data_from_csv():
    """Load data from CSV (original implementation as fallback)"""
    df = pd.read_csv('weather_all_cities.csv', encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # Làm sạch cột gốc và giữ nguyên tên, không tạo cột mới
    df['Temp'] = df['Temp'].str.replace(' °c', '').str.replace('°c', '').astype(float)
    df['Rain'] = df['Rain'].str.replace('mm', '').astype(float)
    df['Cloud'] = df['Cloud'].str.replace('%', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' mb', '').astype(float)
    df['Wind'] = df['Wind'].str.replace(' km/h', '').astype(float)
    df['Gust'] = df['Gust'].str.replace(' km/h', '').astype(float)
    
    # Encode cột Dir (hướng gió) thành số
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
    
    # Cyclical encoding (cho improved model)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def create_features_for_prediction(df, target_col='Temp'):
    """Tạo advanced features cho dữ liệu cần predict (giống train_improved_models.py)"""
    df = df.copy()
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_data = df[city_mask].copy().sort_values('datetime').reset_index(drop=True)
        
        # 1. Lag features của temperature (chỉ lag_12, lag_24)
        for lag in [12, 24]:
            df.loc[city_mask, f'{target_col}_lag_{lag}'] = city_data[target_col].shift(lag).values
        
        # 2. Rolling features của temperature (chỉ rolling_12, rolling_24)
        for window in [12, 24]:
            df.loc[city_mask, f'{target_col}_rolling_mean_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'{target_col}_rolling_std_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[city_mask, f'{target_col}_rolling_max_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).max().values
            df.loc[city_mask, f'{target_col}_rolling_min_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).min().values
        
        # 3. Lag features của Pressure, Wind, Cloud
        for lag in [1, 3, 6, 12]:
            df.loc[city_mask, f'Pressure_lag_{lag}'] = city_data['Pressure'].shift(lag).values
            df.loc[city_mask, f'Wind_lag_{lag}'] = city_data['Wind'].shift(lag).values
            df.loc[city_mask, f'Cloud_lag_{lag}'] = city_data['Cloud'].shift(lag).values
        
        # 4. Rolling features của Pressure, Wind, Cloud
        for window in [3, 6, 12, 24]:
            df.loc[city_mask, f'Pressure_rolling_mean_{window}'] = city_data['Pressure'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Wind_rolling_mean_{window}'] = city_data['Wind'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Cloud_rolling_mean_{window}'] = city_data['Cloud'].shift(1).rolling(window=window, min_periods=1).mean().values
        
        # 5. Context features
        city_data['hour'] = city_data['datetime'].dt.hour
        df.loc[city_mask, f'{target_col}_same_hour_1d_ago'] = city_data.groupby('hour')[target_col].shift(8).values
        df.loc[city_mask, f'{target_col}_same_hour_avg_7d'] = city_data.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
    
    # 6. Interaction features
    df['hour_month_interaction'] = df['hour'] * df['month']
    df['pressure_wind_interaction'] = df['Pressure'] * df['Wind'] / 100
    df['cloud_hour_interaction'] = df['Cloud'] * df['hour'] / 100
    
    # 7. City encoding - đảm bảo tất cả các cột city được tạo
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    
    # Xóa các cột city_* cũ nếu có
    city_cols_to_drop = [col for col in df.columns if col.startswith('city_')]
    if city_cols_to_drop:
        df = df.drop(columns=city_cols_to_drop)
    
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    
    # Thêm các cột city còn thiếu với giá trị 0
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    return df


@app.route('/test')
def test():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('forecast.html')

@app.route('/charts/year')
def charts_year():
    return render_template('charts_year.html')

@app.route('/charts/year/all')
def charts_year_all():
    return render_template('charts_year_all.html')

@app.route('/charts/month')
def charts_month():
    return render_template('charts_month.html')

@app.route('/charts/month/all')
def charts_month_all():
    return render_template('charts_month_all.html')

@app.route('/charts/day')
def charts_day():
    return render_template('charts_day.html')

@app.route('/charts/day/all')
def charts_day_all():
    return render_template('charts_day_all.html')

@app.route('/charts/hour')
def charts_hour():
    return render_template('charts_hour.html')

@app.route('/charts/hour/all')
def charts_hour_all():
    return render_template('charts_hour_all.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        date_str = data.get('date')
        city = data.get('city', 'vinh')
        
        try:
            selected_date = pd.to_datetime(date_str).date()
        except:
            return jsonify({'error': 'Invalid date format'}), 400
        
        df = load_data()
        
        train_end = '2022-12-31'
        test_start = '2023-01-01'
        test_df = df[df['datetime'] >= test_start].copy()
        
        # Lấy dữ liệu cho ngày cần predict và 2 ngày trước để tạo features (cần cho lag_24)
        start_date_for_features = pd.to_datetime(selected_date) - pd.Timedelta(days=2)
        data_for_features = test_df[
            (test_df['datetime'] >= start_date_for_features) & 
            (test_df['datetime'] <= pd.to_datetime(selected_date) + pd.Timedelta(days=1)) &
            (test_df['city'] == city)
        ].copy().sort_values('datetime')
        
        if len(data_for_features) == 0:
            return jsonify({'error': f'No data found for {city} on {selected_date}'}), 400
        
        # Tạo advanced features cho temperature
        data_for_features = create_features_for_prediction(data_for_features, 'Temp')
        
        # Lấy dữ liệu cho ngày cần predict
        day_data = data_for_features[
            data_for_features['datetime'].dt.date == selected_date
        ].copy().sort_values('datetime')
        
        if len(day_data) == 0:
            return jsonify({'error': f'No data found for {city} on {selected_date}'}), 400
        
        attr_names = {
            'Temp_numeric': 'Nhiệt độ',
            'Rain_numeric': 'Mưa',
            'Cloud_numeric': 'Mây',
            'Pressure_numeric': 'Áp suất',
            'Wind_numeric': 'Gió',
            'Gust_numeric': 'Gió giật'
        }
        
        units = {
            'Temp_numeric': '°C',
            'Rain_numeric': 'mm',
            'Cloud_numeric': '%',
            'Pressure_numeric': 'mb',
            'Wind_numeric': 'km/h',
            'Gust_numeric': 'km/h'
        }
        
        result = {
            'date': selected_date.strftime('%Y-%m-%d'),
            'city': city,
            'hours': [t.strftime('%H:%M') for t in day_data['datetime']],
            'attributes': {}
        }
        
        for target in target_vars:
            # Chọn model phù hợp: model riêng cho HCM hoặc model chung
            if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                model_key = 'Temp_numeric_hcm'
            else:
                model_key = target
            
            if model_key not in models:
                continue
            
            feature_cols = feature_cols_dict[model_key]
            model = models[model_key]
            
            # Map tên cột từ _numeric sang tên gốc nếu cần (models dùng _numeric, data dùng tên gốc)
            feature_mapping = {}
            for col in feature_cols:
                if col.endswith('_numeric') and col.replace('_numeric', '') in day_data.columns:
                    feature_mapping[col] = col.replace('_numeric', '')
                elif col in day_data.columns:
                    feature_mapping[col] = col
            
            # Tạo X với tên cột đúng
            X = pd.DataFrame()
            for model_col, data_col in feature_mapping.items():
                if data_col in day_data.columns:
                    X[model_col] = day_data[data_col]
                else:
                    X[model_col] = 0
            
            # Đảm bảo tất cả features có trong X
            missing_features = [f for f in feature_cols if f not in X.columns]
            if missing_features:
                for f in missing_features:
                    X[f] = 0
            
            X = X[feature_cols].fillna(0)
            predictions = model.predict(X)
            
            # Map target name
            target_col = target.replace('_numeric', '') if target.endswith('_numeric') else target
            actuals = day_data[target_col].values if target_col in day_data.columns else day_data[target].values
            errors = np.abs(actuals - predictions)
            
            result['attributes'][target.replace('_numeric', '')] = {
                'name': attr_names[target],
                'unit': units[target],
                'actual': actuals.tolist(),
                'predicted': predictions.tolist(),
                'error': errors.tolist(),
                'mae': float(np.mean(errors))
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detail_data', methods=['POST'])
def get_detail_data():
    try:
        data = request.json
        date_str = data.get('date')
        city = data.get('city', 'vinh')
        
        try:
            selected_date = pd.to_datetime(date_str).date()
        except:
            return jsonify({'error': 'Invalid date format'}), 400
        
        df = load_data()
        
        train_end = '2022-12-31'
        test_start = '2023-01-01'
        test_df = df[df['datetime'] >= test_start].copy()
        
        # Lấy dữ liệu cho ngày cần predict và 2 ngày trước để tạo features
        start_date_for_features = pd.to_datetime(selected_date) - pd.Timedelta(days=2)
        data_for_features = test_df[
            (test_df['datetime'] >= start_date_for_features) & 
            (test_df['datetime'] <= pd.to_datetime(selected_date) + pd.Timedelta(days=1)) &
            (test_df['city'] == city)
        ].copy().sort_values('datetime')
        
        if len(data_for_features) == 0:
            return jsonify({'error': f'No data found for {city} on {selected_date}'}), 400
        
        # Tạo advanced features cho temperature
        data_for_features = create_features_for_prediction(data_for_features, 'Temp')
        
        # Lấy dữ liệu cho ngày cần predict
        day_data = data_for_features[
            data_for_features['datetime'].dt.date == selected_date
        ].copy().sort_values('datetime')
        
        if len(day_data) == 0:
            return jsonify({'error': f'No data found for {city} on {selected_date}'}), 400
        
        attr_names = {
            'Temp_numeric': 'Nhiệt độ',
            'Rain_numeric': 'Mưa',
            'Cloud_numeric': 'Mây',
            'Pressure_numeric': 'Áp suất',
            'Wind_numeric': 'Gió',
            'Gust_numeric': 'Gió giật'
        }
        
        units = {
            'Temp_numeric': '°C',
            'Rain_numeric': 'mm',
            'Cloud_numeric': '%',
            'Pressure_numeric': 'mb',
            'Wind_numeric': 'km/h',
            'Gust_numeric': 'km/h'
        }
        
        predictions_dict = {}
        actuals_dict = {}
        min_max = {}
        
        for target in target_vars:
            # Chọn model phù hợp: model riêng cho HCM hoặc model chung
            if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                model_key = 'Temp_numeric_hcm'
            else:
                model_key = target
            
            if model_key not in models:
                continue
            
            feature_cols = feature_cols_dict[model_key]
            model = models[model_key]
            
            # Map tên cột từ _numeric sang tên gốc nếu cần (models dùng _numeric, data dùng tên gốc)
            feature_mapping = {}
            for col in feature_cols:
                if col.endswith('_numeric') and col.replace('_numeric', '') in day_data.columns:
                    feature_mapping[col] = col.replace('_numeric', '')
                elif col in day_data.columns:
                    feature_mapping[col] = col
            
            # Tạo X với tên cột đúng
            X = pd.DataFrame()
            for model_col, data_col in feature_mapping.items():
                if data_col in day_data.columns:
                    X[model_col] = day_data[data_col]
                else:
                    X[model_col] = 0
            
            # Đảm bảo tất cả features có trong X
            missing_features = [f for f in feature_cols if f not in X.columns]
            if missing_features:
                for f in missing_features:
                    X[f] = 0
            
            X = X[feature_cols].fillna(0)
            predictions = model.predict(X)
            
            # Map target name
            target_col = target.replace('_numeric', '') if target.endswith('_numeric') else target
            actuals = day_data[target_col].values if target_col in day_data.columns else day_data[target].values
            
            predictions_dict[target] = predictions
            actuals_dict[target] = actuals
            
            attr_key = target.replace('_numeric', '')
            min_max[attr_key] = {
                'actual_min': float(np.min(actuals)),
                'actual_max': float(np.max(actuals)),
                'predicted_min': float(np.min(predictions)),
                'predicted_max': float(np.max(predictions)),
                'min_error': float(np.abs(np.min(actuals) - np.min(predictions))),
                'max_error': float(np.abs(np.max(actuals) - np.max(predictions)))
            }
        
        result = {
            'date': selected_date.strftime('%Y-%m-%d'),
            'city': city,
            'hours': [t.strftime('%H:%M') for t in day_data['datetime']],
            'attributes': {},
            'min_max': min_max
        }
        
        for target in target_vars:
            if target not in models:
                continue
            
            attr_key = target.replace('_numeric', '')
            predictions = predictions_dict[target]
            actuals = actuals_dict[target]
            errors = np.abs(actuals - predictions)
            
            result['attributes'][attr_key] = {
                'name': attr_names[target],
                'unit': units[target],
                'actual': actuals.tolist(),
                'predicted': predictions.tolist(),
                'error': errors.tolist(),
                'mae': float(np.mean(errors))
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/year', methods=['POST'])
def api_charts_year():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        
        df = load_data()
        city_df = df[df['city'] == city].copy()
        
        # Chỉ hiển thị dữ liệu đến năm 2025
        city_df = city_df[city_df['year'] <= 2025].copy()
        years = sorted(city_df['year'].unique())
        
        result = {
            'years': [int(y) for y in years],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            },
            'predicted': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for year in years:
            year_data = city_df[city_df['year'] == year]
            
            result['actual']['temp'].append(float(year_data['Temp'].mean()))
            result['actual']['cloud'].append(float(year_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(year_data['Pressure'].mean()))
            result['actual']['wind'].append(float(year_data['Wind'].mean()))
            result['actual']['gust'].append(float(year_data['Gust'].mean()))
            
            if year >= 2023:
                predictions = []
                for target in ['Temp', 'Cloud', 'Pressure', 'Wind', 'Gust']:
                    target_key = f'{target}_numeric'
                    # Chọn model phù hợp: model riêng cho HCM hoặc model chung
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in models:
                        year_data_features = create_features_for_prediction(year_data.copy(), target)
                        feature_cols = feature_cols_dict.get(model_key, [])
                        
                        if len(feature_cols) > 0:
                            # Map tên cột từ _numeric sang tên gốc nếu cần
                            feature_mapping = {}
                            for col in feature_cols:
                                if col.endswith('_numeric') and col.replace('_numeric', '') in year_data_features.columns:
                                    feature_mapping[col] = col.replace('_numeric', '')
                                elif col in year_data_features.columns:
                                    feature_mapping[col] = col
                            
                            # Tạo X với tên cột đúng
                            X = pd.DataFrame()
                            for model_col, data_col in feature_mapping.items():
                                if data_col in year_data_features.columns:
                                    X[model_col] = year_data_features[data_col]
                                else:
                                    X[model_col] = 0
                            
                            # Đảm bảo tất cả features có trong X
                            missing_features = [f for f in feature_cols if f not in X.columns]
                            if missing_features:
                                for f in missing_features:
                                    X[f] = 0
                            
                            X = X[feature_cols].fillna(0)
                            pred = models[model_key].predict(X)
                            predictions.append(float(np.mean(pred)))
                        else:
                            predictions.append(None)
                    else:
                        predictions.append(None)
                
                result['predicted']['temp'].append(float(predictions[0]) if predictions[0] is not None else None)
                result['predicted']['cloud'].append(float(predictions[1]) if predictions[1] is not None else None)
                result['predicted']['pressure'].append(float(predictions[2]) if predictions[2] is not None else None)
                result['predicted']['wind'].append(float(predictions[3]) if predictions[3] is not None else None)
                result['predicted']['gust'].append(float(predictions[4]) if predictions[4] is not None else None)
            else:
                result['predicted']['temp'].append(None)
                result['predicted']['cloud'].append(None)
                result['predicted']['pressure'].append(None)
                result['predicted']['wind'].append(None)
                result['predicted']['gust'].append(None)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/year/all', methods=['POST'])
def api_charts_year_all():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        
        df = load_data()
        city_df = df[df['city'] == city].copy()
        
        # Sử dụng toàn bộ dữ liệu (không filter năm)
        years = sorted(city_df['year'].unique())
        
        # Chỉ trả về dữ liệu thực, không có dự đoán
        result = {
            'years': [int(y) for y in years],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for year in years:
            year_data = city_df[city_df['year'] == year]
            
            result['actual']['temp'].append(float(year_data['Temp'].mean()))
            result['actual']['cloud'].append(float(year_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(year_data['Pressure'].mean()))
            result['actual']['wind'].append(float(year_data['Wind'].mean()))
            result['actual']['gust'].append(float(year_data['Gust'].mean()))
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/month', methods=['POST'])
def api_charts_month():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}'}), 400
        
        months = sorted(city_df['month'].unique())
        
        result = {
            'months': [int(m) for m in months],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            },
            'predicted': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for month in months:
            month_data = city_df[city_df['month'] == month]
            
            result['actual']['temp'].append(float(month_data['Temp'].mean()))
            result['actual']['cloud'].append(float(month_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(month_data['Pressure'].mean()))
            result['actual']['wind'].append(float(month_data['Wind'].mean()))
            result['actual']['gust'].append(float(month_data['Gust'].mean()))
            
            if year >= 2023:
                predictions = []
                for target in ['Temp', 'Cloud', 'Pressure', 'Wind', 'Gust']:
                    target_key = f'{target}_numeric'
                    # Chọn model phù hợp: model riêng cho HCM hoặc model chung
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in models:
                        month_data_features = create_features_for_prediction(month_data.copy(), target)
                        feature_cols = feature_cols_dict.get(model_key, [])
                        
                        if len(feature_cols) > 0:
                            # Map tên cột từ _numeric sang tên gốc nếu cần
                            feature_mapping = {}
                            for col in feature_cols:
                                if col.endswith('_numeric') and col.replace('_numeric', '') in month_data_features.columns:
                                    feature_mapping[col] = col.replace('_numeric', '')
                                elif col in month_data_features.columns:
                                    feature_mapping[col] = col
                            
                            # Tạo X với tên cột đúng
                            X = pd.DataFrame()
                            for model_col, data_col in feature_mapping.items():
                                if data_col in month_data_features.columns:
                                    X[model_col] = month_data_features[data_col]
                                else:
                                    X[model_col] = 0
                            
                            # Đảm bảo tất cả features có trong X
                            missing_features = [f for f in feature_cols if f not in X.columns]
                            if missing_features:
                                for f in missing_features:
                                    X[f] = 0
                            
                            X = X[feature_cols].fillna(0)
                            pred = models[model_key].predict(X)
                            predictions.append(float(np.mean(pred)))
                        else:
                            predictions.append(None)
                    else:
                        predictions.append(None)
                
                result['predicted']['temp'].append(float(predictions[0]) if predictions[0] is not None else None)
                result['predicted']['cloud'].append(float(predictions[1]) if predictions[1] is not None else None)
                result['predicted']['pressure'].append(float(predictions[2]) if predictions[2] is not None else None)
                result['predicted']['wind'].append(float(predictions[3]) if predictions[3] is not None else None)
                result['predicted']['gust'].append(float(predictions[4]) if predictions[4] is not None else None)
            else:
                result['predicted']['temp'].append(None)
                result['predicted']['cloud'].append(None)
                result['predicted']['pressure'].append(None)
                result['predicted']['wind'].append(None)
                result['predicted']['gust'].append(None)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/month/all', methods=['POST'])
def api_charts_month_all():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}'}), 400
        
        months = sorted(city_df['month'].unique())
        
        # Chỉ trả về dữ liệu thực, không có dự đoán
        result = {
            'months': [int(m) for m in months],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for month in months:
            month_data = city_df[city_df['month'] == month]
            
            result['actual']['temp'].append(float(month_data['Temp'].mean()))
            result['actual']['cloud'].append(float(month_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(month_data['Pressure'].mean()))
            result['actual']['wind'].append(float(month_data['Wind'].mean()))
            result['actual']['gust'].append(float(month_data['Gust'].mean()))
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/month/all/minmax', methods=['POST'])
def api_charts_month_all_minmax():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        metric = data.get('metric', 'temp')  # temp, cloud, pressure, wind, gust
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}'}), 400
        
        # Map metric name to column name
        metric_map = {
            'temp': 'Temp',
            'cloud': 'Cloud',
            'pressure': 'Pressure',
            'wind': 'Wind',
            'gust': 'Gust'
        }
        
        if metric not in metric_map:
            return jsonify({'error': f'Invalid metric: {metric}'}), 400
        
        col_name = metric_map[metric]
        
        # Find min and max
        min_idx = city_df[col_name].idxmin()
        max_idx = city_df[col_name].idxmax()
        
        min_row = city_df.loc[min_idx]
        max_row = city_df.loc[max_idx]
        
        result = {
            'min': {
                'date': min_row['date'],
                'time': min_row['Time'],
                'value': float(min_row[col_name]),
                'month': int(min_row['month']),
                'day': int(min_row['day'])
            },
            'max': {
                'date': max_row['date'],
                'time': max_row['Time'],
                'value': float(max_row[col_name]),
                'month': int(max_row['month']),
                'day': int(max_row['day'])
            }
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/day', methods=['POST'])
def api_charts_day():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year) & (df['month'] == month)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}-{month:02d}'}), 400
        
        days = sorted(city_df['day'].unique())
        
        result = {
            'days': [int(d) for d in days],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            },
            'predicted': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for day in days:
            day_data = city_df[city_df['day'] == day]
            
            result['actual']['temp'].append(float(day_data['Temp'].mean()))
            result['actual']['cloud'].append(float(day_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(day_data['Pressure'].mean()))
            result['actual']['wind'].append(float(day_data['Wind'].mean()))
            result['actual']['gust'].append(float(day_data['Gust'].mean()))
            
            if year >= 2023:
                predictions = []
                for target in ['Temp', 'Cloud', 'Pressure', 'Wind', 'Gust']:
                    target_key = f'{target}_numeric'
                    # Chọn model phù hợp: model riêng cho HCM hoặc model chung
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in models:
                        day_data_features = create_features_for_prediction(day_data.copy(), target)
                        feature_cols = feature_cols_dict.get(model_key, [])
                        
                        if len(feature_cols) > 0:
                            # Map tên cột từ _numeric sang tên gốc nếu cần
                            feature_mapping = {}
                            for col in feature_cols:
                                if col.endswith('_numeric') and col.replace('_numeric', '') in day_data_features.columns:
                                    feature_mapping[col] = col.replace('_numeric', '')
                                elif col in day_data_features.columns:
                                    feature_mapping[col] = col
                            
                            # Tạo X với tên cột đúng
                            X = pd.DataFrame()
                            for model_col, data_col in feature_mapping.items():
                                if data_col in day_data_features.columns:
                                    X[model_col] = day_data_features[data_col]
                                else:
                                    X[model_col] = 0
                            
                            # Đảm bảo tất cả features có trong X
                            missing_features = [f for f in feature_cols if f not in X.columns]
                            if missing_features:
                                for f in missing_features:
                                    X[f] = 0
                            
                            X = X[feature_cols].fillna(0)
                            pred = models[model_key].predict(X)
                            predictions.append(float(np.mean(pred)))
                        else:
                            predictions.append(None)
                    else:
                        predictions.append(None)
                
                result['predicted']['temp'].append(float(predictions[0]) if predictions[0] is not None else None)
                result['predicted']['cloud'].append(float(predictions[1]) if predictions[1] is not None else None)
                result['predicted']['pressure'].append(float(predictions[2]) if predictions[2] is not None else None)
                result['predicted']['wind'].append(float(predictions[3]) if predictions[3] is not None else None)
                result['predicted']['gust'].append(float(predictions[4]) if predictions[4] is not None else None)
            else:
                result['predicted']['temp'].append(None)
                result['predicted']['cloud'].append(None)
                result['predicted']['pressure'].append(None)
                result['predicted']['wind'].append(None)
                result['predicted']['gust'].append(None)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/day/all', methods=['POST'])
def api_charts_day_all():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year) & (df['month'] == month)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}-{month:02d}'}), 400
        
        days = sorted(city_df['day'].unique())
        
        # Chỉ trả về dữ liệu thực, không có dự đoán
        result = {
            'days': [int(d) for d in days],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for day in days:
            day_data = city_df[city_df['day'] == day]
            
            result['actual']['temp'].append(float(day_data['Temp'].mean()))
            result['actual']['cloud'].append(float(day_data['Cloud'].mean()))
            result['actual']['pressure'].append(float(day_data['Pressure'].mean()))
            result['actual']['wind'].append(float(day_data['Wind'].mean()))
            result['actual']['gust'].append(float(day_data['Gust'].mean()))
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/day/all/minmax', methods=['POST'])
def api_charts_day_all_minmax():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        metric = data.get('metric', 'temp')
        
        df = load_data()
        city_df = df[(df['city'] == city) & (df['year'] == year) & (df['month'] == month)].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} in {year}-{month:02d}'}), 400
        
        metric_map = {
            'temp': 'Temp',
            'cloud': 'Cloud',
            'pressure': 'Pressure',
            'wind': 'Wind',
            'gust': 'Gust'
        }
        
        if metric not in metric_map:
            return jsonify({'error': f'Invalid metric: {metric}'}), 400
        
        col_name = metric_map[metric]
        
        # Tính min/max cho từng ngày trong tháng
        days = sorted(city_df['day'].unique())
        daily_data = []
        
        for day in days:
            day_data = city_df[city_df['day'] == day]
            day_min_idx = day_data[col_name].idxmin()
            day_max_idx = day_data[col_name].idxmax()
            
            day_min_row = day_data.loc[day_min_idx]
            day_max_row = day_data.loc[day_max_idx]
            
            daily_data.append({
                'day': int(day),
                'min': {
                    'value': float(day_min_row[col_name]),
                    'time': day_min_row['Time'],
                    'date': day_min_row['date']
                },
                'max': {
                    'value': float(day_max_row[col_name]),
                    'time': day_max_row['Time'],
                    'date': day_max_row['date']
                }
            })
        
        # Tìm ngày có giá trị cao nhất và thấp nhất cho tất cả các metric
        all_max_values = [d['max']['value'] for d in daily_data]
        all_min_values = [d['min']['value'] for d in daily_data]
        
        highest_day_idx = all_max_values.index(max(all_max_values))
        lowest_day_idx = all_min_values.index(min(all_min_values))
        
        highest_day = daily_data[highest_day_idx]['day']
        lowest_day = daily_data[lowest_day_idx]['day']
        
        result = {
            'daily_data': daily_data,
            'highest_day': highest_day,
            'lowest_day': lowest_day
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/hour/all', methods=['POST'])
def api_charts_hour_all():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        day = int(data.get('day', 1))
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] == year) & 
            (df['month'] == month) & 
            (df['day'] == day)
        ].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} on {year}-{month:02d}-{day:02d}'}), 400
        
        hours = sorted(city_df['hour'].unique())
        
        # Chỉ trả về dữ liệu thực, không có dự đoán
        result = {
            'hours': [int(h) for h in hours],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for hour in hours:
            hour_data = city_df[city_df['hour'] == hour]
            
            if len(hour_data) > 0:
                result['actual']['temp'].append(float(hour_data['Temp'].iloc[0]))
                result['actual']['cloud'].append(float(hour_data['Cloud'].iloc[0]))
                result['actual']['pressure'].append(float(hour_data['Pressure'].iloc[0]))
                result['actual']['wind'].append(float(hour_data['Wind'].iloc[0]))
                result['actual']['gust'].append(float(hour_data['Gust'].iloc[0]))
            else:
                result['actual']['temp'].append(None)
                result['actual']['cloud'].append(None)
                result['actual']['pressure'].append(None)
                result['actual']['wind'].append(None)
                result['actual']['gust'].append(None)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/hour/all/minmax', methods=['POST'])
def api_charts_hour_all_minmax():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        day = int(data.get('day', 1))
        metric = data.get('metric', 'temp')
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] == year) & 
            (df['month'] == month) & 
            (df['day'] == day)
        ].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} on {year}-{month:02d}-{day:02d}'}), 400
        
        metric_map = {
            'temp': 'Temp',
            'cloud': 'Cloud',
            'pressure': 'Pressure',
            'wind': 'Wind',
            'gust': 'Gust'
        }
        
        if metric not in metric_map:
            return jsonify({'error': f'Invalid metric: {metric}'}), 400
        
        col_name = metric_map[metric]
        
        # Tính min/max cho từng giờ trong ngày
        hours = sorted(city_df['hour'].unique())
        hourly_data = []
        
        for hour in hours:
            hour_data = city_df[city_df['hour'] == hour]
            
            if len(hour_data) > 0:
                hour_min_idx = hour_data[col_name].idxmin()
                hour_max_idx = hour_data[col_name].idxmax()
                
                hour_min_row = hour_data.loc[hour_min_idx]
                hour_max_row = hour_data.loc[hour_max_idx]
                
                hourly_data.append({
                    'hour': int(hour),
                    'min': {
                        'value': float(hour_min_row[col_name]),
                        'time': hour_min_row['Time'],
                        'date': hour_min_row['date']
                    },
                    'max': {
                        'value': float(hour_max_row[col_name]),
                        'time': hour_max_row['Time'],
                        'date': hour_max_row['date']
                    }
                })
        
        # Tìm giờ có giá trị cao nhất và thấp nhất cho tất cả các metric
        all_max_values = [d['max']['value'] for d in hourly_data]
        all_min_values = [d['min']['value'] for d in hourly_data]
        
        highest_hour_idx = all_max_values.index(max(all_max_values))
        lowest_hour_idx = all_min_values.index(min(all_min_values))
        
        highest_hour = hourly_data[highest_hour_idx]['hour']
        lowest_hour = hourly_data[lowest_hour_idx]['hour']
        
        result = {
            'hourly_data': hourly_data,
            'highest_hour': highest_hour,
            'lowest_hour': lowest_hour
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/hour', methods=['POST'])
def api_charts_hour():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year = int(data.get('year', 2023))
        month = int(data.get('month', 1))
        day = int(data.get('day', 1))
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] == year) & 
            (df['month'] == month) & 
            (df['day'] == day)
        ].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} on {year}-{month:02d}-{day:02d}'}), 400
        
        hours = sorted(city_df['hour'].unique())
        
        result = {
            'hours': [int(h) for h in hours],
            'actual': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            },
            'predicted': {
                'temp': [],
                'cloud': [],
                'pressure': [],
                'wind': [],
                'gust': []
            }
        }
        
        for hour in hours:
            hour_data = city_df[city_df['hour'] == hour]
            
            if len(hour_data) > 0:
                result['actual']['temp'].append(float(hour_data['Temp'].iloc[0]))
                result['actual']['cloud'].append(float(hour_data['Cloud'].iloc[0]))
                result['actual']['pressure'].append(float(hour_data['Pressure'].iloc[0]))
                result['actual']['wind'].append(float(hour_data['Wind'].iloc[0]))
                result['actual']['gust'].append(float(hour_data['Gust'].iloc[0]))
            else:
                result['actual']['temp'].append(None)
                result['actual']['cloud'].append(None)
                result['actual']['pressure'].append(None)
                result['actual']['wind'].append(None)
                result['actual']['gust'].append(None)
            
            if year >= 2023 and len(hour_data) > 0:
                predictions = []
                for target in ['Temp', 'Cloud', 'Pressure', 'Wind', 'Gust']:
                    target_key = f'{target}_numeric'
                    # Chọn model phù hợp: model riêng cho HCM hoặc model chung
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in models:
                        hour_data_features = create_features_for_prediction(hour_data.copy(), target)
                        feature_cols = feature_cols_dict.get(model_key, [])
                        
                        if len(feature_cols) > 0:
                            # Map tên cột từ _numeric sang tên gốc nếu cần
                            feature_mapping = {}
                            for col in feature_cols:
                                if col.endswith('_numeric') and col.replace('_numeric', '') in hour_data_features.columns:
                                    feature_mapping[col] = col.replace('_numeric', '')
                                elif col in hour_data_features.columns:
                                    feature_mapping[col] = col
                            
                            # Tạo X với tên cột đúng
                            X = pd.DataFrame()
                            for model_col, data_col in feature_mapping.items():
                                if data_col in hour_data_features.columns:
                                    X[model_col] = hour_data_features[data_col]
                                else:
                                    X[model_col] = 0
                            
                            # Đảm bảo tất cả features có trong X
                            missing_features = [f for f in feature_cols if f not in X.columns]
                            if missing_features:
                                for f in missing_features:
                                    X[f] = 0
                            
                            X = X[feature_cols].fillna(0)
                            pred = models[model_key].predict(X)
                            predictions.append(float(pred[0]) if len(pred) > 0 else None)
                        else:
                            predictions.append(None)
                    else:
                        predictions.append(None)
                
                result['predicted']['temp'].append(float(predictions[0]) if predictions[0] is not None else None)
                result['predicted']['cloud'].append(float(predictions[1]) if predictions[1] is not None else None)
                result['predicted']['pressure'].append(float(predictions[2]) if predictions[2] is not None else None)
                result['predicted']['wind'].append(float(predictions[3]) if predictions[3] is not None else None)
                result['predicted']['gust'].append(float(predictions[4]) if predictions[4] is not None else None)
            else:
                result['predicted']['temp'].append(None)
                result['predicted']['cloud'].append(None)
                result['predicted']['pressure'].append(None)
                result['predicted']['wind'].append(None)
                result['predicted']['gust'].append(None)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/accuracy_data')
def accuracy_data():
    try:
        from database import get_latest_accuracy_results
        
        result = get_latest_accuracy_results()
        
        if result is None:
            accuracy_file = 'accuracy_results.json'
            if os.path.exists(accuracy_file):
                with open(accuracy_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            else:
                return jsonify({
                    'error': 'Không tìm thấy dữ liệu accuracy. Vui lòng chạy: python calculate_accuracy.py'
                }), 404
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        
        df = load_data()
        
        # Lấy ngày hiện tại
        today = pd.Timestamp.now().date()
        
        # Lấy dữ liệu mới nhất của thành phố
        city_df = df[df['city'] == city].copy().sort_values('datetime')
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city}'}), 400
        
        # Lấy ngày cuối cùng có dữ liệu
        last_date = city_df['datetime'].dt.date.max()
        
        # Nếu hôm nay <= ngày cuối cùng có dữ liệu, thì dự báo từ ngày mai
        # Nếu hôm nay > ngày cuối cùng có dữ liệu, thì dự báo từ hôm nay
        if today <= last_date:
            # Dữ liệu đã có đến hôm nay hoặc sau hôm nay, dự báo từ ngày mai
            start_forecast_date = last_date + pd.Timedelta(days=1)
        else:
            # Dữ liệu chưa có đến hôm nay, dự báo từ hôm nay
            start_forecast_date = today
        
        # Tạo danh sách 7 ngày (từ start_forecast_date + 6 ngày tiếp theo)
        forecast_dates = [start_forecast_date + pd.Timedelta(days=i) for i in range(7)]
        
        result = {
            'city': city,
            'today': today.strftime('%Y-%m-%d'),
            'last_data_date': last_date.strftime('%Y-%m-%d'),
            'start_forecast_date': start_forecast_date.strftime('%Y-%m-%d'),
            'forecast': []
        }
        
        # Lấy dữ liệu quá khứ để tạo features (cần ít nhất 2 ngày trước)
        start_date_for_features = forecast_dates[0] - pd.Timedelta(days=2)
        end_date_for_features = forecast_dates[-1]
        
        # Lấy dữ liệu có sẵn (từ quá khứ đến ngày cuối cùng có dữ liệu)
        available_data = city_df[
            (city_df['datetime'].dt.date >= start_date_for_features) & 
            (city_df['datetime'].dt.date <= last_date)
        ].copy().sort_values('datetime')
        
        # Tạo features cho dữ liệu có sẵn
        if len(available_data) > 0:
            available_data = create_features_for_prediction(available_data, 'Temp')
        
        # Dự báo cho từng ngày
        for forecast_date in forecast_dates:
            day_forecast = {
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day_name': forecast_date.strftime('%A'),
                'is_today': forecast_date == today,
                'is_first_day': forecast_date == start_forecast_date,
                'attributes': {}
            }
            
            # Tạo dữ liệu cho 24 giờ trong ngày (mỗi 3 giờ: 0, 3, 6, 9, 12, 15, 18, 21)
            hours = [0, 3, 6, 9, 12, 15, 18, 21]
            
            # Tạo DataFrame cho ngày dự báo
            forecast_data_list = []
            for hour in hours:
                dt = pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=hour)
                
                # Lấy dữ liệu gần nhất từ available_data hoặc dùng giá trị trung bình
                if len(available_data) > 0:
                    # Tìm dữ liệu cùng giờ gần nhất
                    same_hour_data = available_data[
                        (available_data['datetime'].dt.hour == hour) & 
                        (available_data['datetime'].dt.date <= forecast_date)
                    ]
                    
                    if len(same_hour_data) > 0:
                        latest_row = same_hour_data.iloc[-1].copy()
                    else:
                        # Lấy dữ liệu gần nhất
                        latest_row = available_data.iloc[-1].copy()
                else:
                    # Tạo row mặc định
                    latest_row = pd.Series()
                
                # Tạo row mới cho giờ này
                row = {
                    'datetime': dt,
                    'city': city,
                    'year': forecast_date.year,
                    'month': forecast_date.month,
                    'day': forecast_date.day,
                    'hour': hour,
                    'day_of_year': forecast_date.timetuple().tm_yday,
                    'day_of_week': forecast_date.weekday(),
                    'is_weekend': 1 if forecast_date.weekday() >= 5 else 0,
                }
                
                # Tính season
                if forecast_date.month in [12, 1, 2]:
                    row['season'] = 0
                elif forecast_date.month in [3, 4, 5]:
                    row['season'] = 1
                elif forecast_date.month in [6, 7, 8]:
                    row['season'] = 2
                else:
                    row['season'] = 3
                
                # Cyclical encoding
                row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                row['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
                row['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
                row['day_of_year_sin'] = np.sin(2 * np.pi * row['day_of_year'] / 365)
                row['day_of_year_cos'] = np.cos(2 * np.pi * row['day_of_year'] / 365)
                
                # Copy các giá trị từ latest_row nếu có
                for col in ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']:
                    if col in latest_row.index and pd.notna(latest_row[col]):
                        row[col] = latest_row[col]
                    else:
                        # Giá trị mặc định
                        defaults = {'Temp': 25.0, 'Rain': 0.0, 'Cloud': 50.0, 
                                   'Pressure': 1013.0, 'Wind': 10.0, 'Gust': 15.0, 'Dir': 0}
                        row[col] = defaults.get(col, 0.0)
                
                # Copy các features từ latest_row nếu có
                for col in available_data.columns:
                    if col not in row and col.startswith(('Temp_', 'Pressure_', 'Wind_', 'Cloud_')):
                        if col in latest_row.index and pd.notna(latest_row[col]):
                            row[col] = latest_row[col]
                        else:
                            row[col] = 0.0
                
                forecast_data_list.append(row)
            
            forecast_df = pd.DataFrame(forecast_data_list)
            
            # Tạo features cho forecast_df
            if len(available_data) > 0:
                # Kết hợp available_data và forecast_df để tính features
                combined_df = pd.concat([available_data, forecast_df], ignore_index=True).sort_values('datetime')
                combined_df = create_features_for_prediction(combined_df, 'Temp')
                
                # Lấy lại phần forecast
                forecast_df = combined_df[combined_df['datetime'].dt.date == forecast_date].copy()
            else:
                forecast_df = create_features_for_prediction(forecast_df, 'Temp')
            
            # Predict cho từng attribute
            all_predictions = {}
            for target in target_vars:
                # Chọn model phù hợp
                if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in models:
                    model_key = 'Temp_numeric_hcm'
                else:
                    model_key = target
                
                if model_key not in models:
                    continue
                
                feature_cols = feature_cols_dict[model_key]
                model = models[model_key]
                
                # Map tên cột
                feature_mapping = {}
                for col in feature_cols:
                    if col.endswith('_numeric') and col.replace('_numeric', '') in forecast_df.columns:
                        feature_mapping[col] = col.replace('_numeric', '')
                    elif col in forecast_df.columns:
                        feature_mapping[col] = col
                
                # Tạo X
                X = pd.DataFrame()
                for model_col, data_col in feature_mapping.items():
                    if data_col in forecast_df.columns:
                        X[model_col] = forecast_df[data_col]
                    else:
                        X[model_col] = 0
                
                missing_features = [f for f in feature_cols if f not in X.columns]
                if missing_features:
                    for f in missing_features:
                        X[f] = 0
                
                X = X[feature_cols].fillna(0)
                predictions = model.predict(X)
                
                target_name = target.replace('_numeric', '')
                day_forecast['attributes'][target_name] = {
                    'hourly': [float(p) for p in predictions],
                    'min': float(np.min(predictions)),
                    'max': float(np.max(predictions)),
                    'avg': float(np.mean(predictions))
                }
                
                # Lưu tất cả predictions để dùng cho ngày tiếp theo
                all_predictions[target_name] = predictions
            
            result['forecast'].append(day_forecast)
            
            # Cập nhật available_data với tất cả predictions để dùng cho ngày tiếp theo
            if len(all_predictions) > 0:
                for idx, hour in enumerate(hours):
                    dt = pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=hour)
                    new_row = forecast_df.iloc[idx].copy()
                    
                    # Cập nhật tất cả các attributes đã predict
                    for attr_name, pred_values in all_predictions.items():
                        if idx < len(pred_values):
                            new_row[attr_name] = float(pred_values[idx])
                    
                    # Thêm vào available_data
                    new_row_df = pd.DataFrame([new_row])
                    available_data = pd.concat([available_data, new_row_df], ignore_index=True)
                
                available_data = available_data.sort_values('datetime').reset_index(drop=True)
                # Tạo lại features sau khi thêm dữ liệu mới
                available_data = create_features_for_prediction(available_data, 'Temp')
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    print("="*70)
    print("INITIALIZING WEATHER FORECASTING WEB APP")
    print("="*70)
    
    if not load_models():
        print("\n❌ Không thể load models. Vui lòng chạy:")
        print("   python train_and_save_models.py")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("WEB APP READY!")
    print("="*70)
    print("\nOpen browser and visit: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop server\n")
    app.run(debug=True, host='127.0.0.1', port=5000)

