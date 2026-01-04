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

models_final = {}
feature_cols_dict_final = {}
models_improved = {}
feature_cols_dict_improved = {}
models = {}
feature_cols_dict = {}
target_vars = ['Temp_numeric', 'Rain_numeric', 'Cloud_numeric', 
               'Pressure_numeric', 'Wind_numeric', 'Gust_numeric']

def load_models():
    """Load cả 2 model: final (để dự báo) và improved (để test/validation)"""
    global models_final, feature_cols_dict_final
    global models_improved, feature_cols_dict_improved
    global models, feature_cols_dict
    
    loaded_final = False
    loaded_improved = False
    
    # Load final model (để dự báo)
    if os.path.exists('weather_models_final.pkl'):
        try:
            print("Loading weather_models_final.pkl (for forecasting)...")
            with open('weather_models_final.pkl', 'rb') as f:
                models_data = pickle.load(f)
            models_final = models_data['models']
            feature_cols_dict_final = models_data['feature_cols']
            loaded_final = True
            print(f"✅ Đã load {len(models_final)} FINAL models thành công!")
            print(f"   Models: {list(models_final.keys())}")
            print(f"   Using FINAL models (trained on ALL data 2017-2026-01-02)")
            if 'Temp_numeric' in models_final:
                print(f"   - Temp model: R² ~97%")
            if 'Temp_numeric_hcm' in models_final:
                print(f"   - HCM Temp model: R² ~97.6%")
        except Exception as e:
            print(f"⚠️  Lỗi khi load weather_models_final.pkl: {e}")
    else:
        print("⚠️  Không tìm thấy weather_models_final.pkl")
    
    # Load improved model (để test/validation)
    if os.path.exists('weather_models_improved.pkl'):
        try:
            print("Loading weather_models_improved.pkl (for test/validation)...")
            with open('weather_models_improved.pkl', 'rb') as f:
                models_data = pickle.load(f)
            models_improved = models_data['models']
            feature_cols_dict_improved = models_data['feature_cols']
            loaded_improved = True
            print(f"✅ Đã load {len(models_improved)} IMPROVED models thành công!")
            print(f"   Models: {list(models_improved.keys())}")
            print(f"   Using IMPROVED models (trained with train/val/test split)")
            if 'Temp_numeric' in models_improved:
                print(f"   - Temp model: R² ~94%")
            if 'Temp_numeric_hcm' in models_improved:
                print(f"   - HCM Temp model: R² ~91.6%")
        except Exception as e:
            print(f"⚠️  Lỗi khi load weather_models_improved.pkl: {e}")
    else:
        print("⚠️  Không tìm thấy weather_models_improved.pkl")
    
    # Set default models (ưu tiên final, fallback improved)
    if loaded_final:
        models = models_final
        feature_cols_dict = feature_cols_dict_final
        print("\n✅ Sử dụng FINAL models làm mặc định (cho dự báo)")
    elif loaded_improved:
        models = models_improved
        feature_cols_dict = feature_cols_dict_improved
        print("\n⚠️  Sử dụng IMPROVED models làm mặc định (fallback)")
    else:
        # Fallback to old model
        if os.path.exists('weather_models.pkl'):
            try:
                print("Loading weather_models.pkl (old model)...")
                with open('weather_models.pkl', 'rb') as f:
                    models_data = pickle.load(f)
                models = models_data['models']
                feature_cols_dict = models_data['feature_cols']
                print(f"✅ Đã load {len(models)} OLD models thành công!")
            except Exception as e:
                print(f"❌ Lỗi khi load weather_models.pkl: {e}")
                return False
        else:
            print(f"❌ Không tìm thấy model nào!")
            print("   Vui lòng chạy: python train_final_models.py hoặc python train_improved_models.py")
            return False
    
    if not loaded_final and not loaded_improved:
        return False
    
    return True

def get_models_for_route(use_improved=False):
    """
    Lấy models và feature_cols phù hợp cho route
    - use_improved=True: dùng improved model (cho test/validation/charts)
    - use_improved=False: dùng final model (cho forecast)
    """
    global models_final, feature_cols_dict_final
    global models_improved, feature_cols_dict_improved
    
    if use_improved and len(models_improved) > 0:
        return models_improved, feature_cols_dict_improved
    elif len(models_final) > 0:
        return models_final, feature_cols_dict_final
    elif len(models_improved) > 0:
        return models_improved, feature_cols_dict_improved
    else:
        return models, feature_cols_dict

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
        
        # Dùng improved model cho test/validation
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
            if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                model_key = 'Temp_numeric_hcm'
            else:
                model_key = target
            
            if model_key not in route_models:
                continue
            
            feature_cols = route_feature_cols[model_key]
            model = route_models[model_key]
            
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
        
        # Dùng improved model cho test/validation
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
            if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                model_key = 'Temp_numeric_hcm'
            else:
                model_key = target
            
            if model_key not in route_models:
                continue
            
            feature_cols = route_feature_cols[model_key]
            model = route_models[model_key]
            
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
            if target not in route_models:
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
        
        # Dùng improved model cho đánh giá dữ liệu
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in route_models:
                        year_data_features = create_features_for_prediction(year_data.copy(), target)
                        feature_cols = route_feature_cols.get(model_key, [])
                        
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
                            pred = route_models[model_key].predict(X)
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
        
        # Dùng improved model cho đánh giá dữ liệu
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in route_models:
                        month_data_features = create_features_for_prediction(month_data.copy(), target)
                        feature_cols = route_feature_cols.get(model_key, [])
                        
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
                            pred = route_models[model_key].predict(X)
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
        
        # Dùng improved model cho đánh giá dữ liệu
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in route_models:
                        day_data_features = create_features_for_prediction(day_data.copy(), target)
                        feature_cols = route_feature_cols.get(model_key, [])
                        
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
                            pred = route_models[model_key].predict(X)
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
        
        # Dùng improved model cho charts (test/validation)
        route_models, route_feature_cols = get_models_for_route(use_improved=True)
        
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
                    if target_key == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                        model_key = 'Temp_numeric_hcm'
                    else:
                        model_key = target_key
                    
                    if model_key in route_models:
                        hour_data_features = create_features_for_prediction(hour_data.copy(), target)
                        feature_cols = route_feature_cols.get(model_key, [])
                        
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
                            pred = route_models[model_key].predict(X)
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
        
        # Lưu predictions của mỗi ngày để dùng cho ngày tiếp theo
        previous_day_predictions = {}  # {date: {attr_name: [predictions for 8 hours]}}
        
        # Dự báo cho từng ngày
        for day_idx, forecast_date in enumerate(forecast_dates):
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
            is_first_day = (day_idx == 0)
            
            for hour in hours:
                dt = pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=hour)
                
                latest_row = pd.Series()
                
                if is_first_day:
                    # Ngày đầu tiên: Dùng dữ liệu thực tế từ database
                    real_data = city_df[city_df['datetime'].dt.date < forecast_date].copy()
                    
                    if len(real_data) > 0:
                        # Tìm dữ liệu cùng giờ gần nhất từ dữ liệu thực tế
                        same_hour_real = real_data[
                            (real_data['datetime'].dt.hour == hour)
                        ]
                        
                        if len(same_hour_real) > 0:
                            # Lấy giá trị trung bình của cùng giờ trong 7 ngày gần nhất (chỉ cột numeric)
                            numeric_cols = same_hour_real.select_dtypes(include=[np.number]).columns
                            latest_row = same_hour_real.tail(7)[numeric_cols].mean()
                            # Thêm các cột non-numeric từ row cuối cùng
                            last_row = same_hour_real.tail(1).iloc[0]
                            for col in same_hour_real.columns:
                                if col not in numeric_cols:
                                    latest_row[col] = last_row[col]
                        else:
                            # Lấy giá trị trung bình của giờ gần nhất (chỉ cột numeric)
                            numeric_cols = real_data.select_dtypes(include=[np.number]).columns
                            latest_row = real_data.tail(24)[numeric_cols].mean()
                            # Thêm các cột non-numeric từ row cuối cùng
                            last_row = real_data.tail(1).iloc[0]
                            for col in real_data.columns:
                                if col not in numeric_cols:
                                    latest_row[col] = last_row[col]
                else:
                    # Ngày sau: Dùng dự đoán của ngày trước
                    previous_date = forecast_dates[day_idx - 1]
                    if previous_date in previous_day_predictions:
                        prev_predictions = previous_day_predictions[previous_date]
                        hour_idx = hours.index(hour) if hour in hours else 0
                        
                        # Lấy dự đoán của giờ tương ứng từ ngày trước
                        if hour_idx < len(hours):
                            for attr_name, pred_values in prev_predictions.items():
                                if hour_idx < len(pred_values):
                                    latest_row[attr_name] = float(pred_values[hour_idx])
                        
                        # Nếu không có đủ dữ liệu, lấy từ giờ cuối cùng của ngày trước
                        if len(latest_row) == 0:
                            last_hour_idx = len(hours) - 1
                            for attr_name, pred_values in prev_predictions.items():
                                if last_hour_idx < len(pred_values):
                                    latest_row[attr_name] = float(pred_values[last_hour_idx])
                    
                    # Nếu vẫn không có dữ liệu, tạo row mặc định
                    if len(latest_row) == 0:
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
                        base_value = float(latest_row[col])
                        
                        # Tính số ngày từ ngày đầu tiên dự báo
                        days_from_start = (forecast_date - start_forecast_date).days
                        
                        # Ngày đầu tiên: Giữ nguyên, không biến đổi
                        if days_from_start == 0:
                            row[col] = base_value
                        else:
                            # Từ ngày 2 trở đi: Áp dụng biến đổi lớn để mỗi ngày khác nhau rõ ràng
                            # Biến đổi dựa trên ngày trong tuần (cuối tuần có thể khác)
                            day_of_week_factor = 1.0
                            if row['day_of_week'] >= 5:  # Cuối tuần
                                if col == 'Temp':
                                    day_of_week_factor = 1.02  # Cuối tuần có thể nóng hơn một chút
                            
                            # Biến đổi dựa trên số ngày - tăng biến đổi lớn hơn
                            # Kết hợp biến đổi tuần hoàn và tuyến tính
                            day_progression_sin = np.sin(2 * np.pi * days_from_start / 7) * 4.0  # Biến đổi ±4 độ (tuần hoàn)
                            day_progression_linear = days_from_start * 0.8  # Biến đổi tuyến tính +0.8 độ mỗi ngày
                            day_progression = day_progression_sin + day_progression_linear
                            
                            # Biến đổi theo mùa với thay đổi lớn hơn
                            season_factor = 1.0
                            if row['season'] == 0:  # Đông
                                if col == 'Temp':
                                    season_factor = 0.96 + (days_from_start % 5) * 0.02  # Biến đổi lớn hơn theo ngày
                            elif row['season'] == 2:  # Hè
                                if col == 'Temp':
                                    season_factor = 1.04 - (days_from_start % 5) * 0.02
                            
                            # Thêm biến đổi ngẫu nhiên nhỏ dựa trên ngày để đảm bảo khác biệt
                            day_random = (days_from_start * 17) % 10 - 5  # Biến đổi -5 đến +5 độ dựa trên ngày
                            
                            # Áp dụng biến đổi
                            if col == 'Temp':
                                # Tổng hợp tất cả biến đổi để đảm bảo mỗi ngày khác nhau rõ ràng
                                row[col] = base_value * season_factor * day_of_week_factor + day_progression + day_random * 0.3
                            elif col == 'Pressure':
                                # Áp suất biến đổi nhỏ hơn
                                pressure_variation = np.sin(2 * np.pi * days_from_start / 5) * 1.5
                                row[col] = base_value + pressure_variation
                            elif col == 'Rain':
                                # Mưa có thể thay đổi đột ngột
                                rain_variation = np.sin(2 * np.pi * days_from_start / 4) * 0.5
                                row[col] = max(0, base_value + rain_variation)
                            elif col == 'Cloud':
                                # Mây biến đổi
                                cloud_variation = np.sin(2 * np.pi * days_from_start / 6) * 5.0
                                row[col] = max(0, min(100, base_value + cloud_variation))
                            elif col in ['Wind', 'Gust']:
                                # Gió biến đổi
                                wind_variation = np.sin(2 * np.pi * days_from_start / 5) * 1.0
                                row[col] = max(0, base_value + wind_variation)
                            else:
                                row[col] = base_value
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
            
            # Đảm bảo forecast_df có đủ các cột cần thiết
            required_cols = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir', 'datetime', 'city']
            for col in required_cols:
                if col not in forecast_df.columns:
                    if col == 'datetime':
                        forecast_df['datetime'] = [pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=h) for h in hours]
                    elif col == 'city':
                        forecast_df['city'] = city
                    else:
                        defaults = {'Temp': 25.0, 'Rain': 0.0, 'Cloud': 50.0, 
                                   'Pressure': 1013.0, 'Wind': 10.0, 'Gust': 15.0, 'Dir': 0}
                        forecast_df[col] = defaults.get(col, 0.0)
            
            # Tạo features cho forecast_df
            if is_first_day:
                # Ngày đầu tiên: Dùng dữ liệu thực tế từ database
                real_data_for_features = city_df[city_df['datetime'].dt.date < forecast_date].tail(100).copy()
                
                if len(real_data_for_features) > 0:
                    try:
                        # Kết hợp dữ liệu thực tế và forecast_df để tính features
                        combined_df = pd.concat([real_data_for_features, forecast_df], ignore_index=True).sort_values('datetime')
                        combined_df = create_features_for_prediction(combined_df, 'Temp')
                        
                        # Lấy lại phần forecast
                        forecast_df = combined_df[combined_df['datetime'].dt.date == forecast_date].copy()
                        
                        # Đảm bảo có đủ rows
                        if len(forecast_df) == 0:
                            raise ValueError("No forecast data after feature creation")
                    except Exception as e:
                        # Nếu có lỗi, fallback về cách đơn giản
                        try:
                            forecast_df = create_features_for_prediction(forecast_df, 'Temp')
                        except:
                            pass
                else:
                    try:
                        forecast_df = create_features_for_prediction(forecast_df, 'Temp')
                    except:
                        pass
            else:
                # Ngày sau: Dùng available_data (đã có predictions của ngày trước)
                if len(available_data) > 0:
                    try:
                        # Kết hợp available_data (có predictions) và forecast_df để tính features
                        combined_df = pd.concat([available_data, forecast_df], ignore_index=True).sort_values('datetime')
                        combined_df = create_features_for_prediction(combined_df, 'Temp')
                        
                        # Lấy lại phần forecast
                        forecast_df = combined_df[combined_df['datetime'].dt.date == forecast_date].copy()
                        
                        # Đảm bảo có đủ rows
                        if len(forecast_df) == 0:
                            raise ValueError("No forecast data after feature creation")
                    except Exception as e:
                        # Nếu có lỗi, fallback về cách đơn giản
                        try:
                            forecast_df = create_features_for_prediction(forecast_df, 'Temp')
                        except:
                            pass
                else:
                    try:
                        forecast_df = create_features_for_prediction(forecast_df, 'Temp')
                    except:
                        pass
            
            # Predict cho từng attribute - dùng final model cho forecast
            route_models, route_feature_cols = get_models_for_route(use_improved=False)
            all_predictions = {}
            for target in target_vars:
                # Chọn model phù hợp
                if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                    model_key = 'Temp_numeric_hcm'
                else:
                    model_key = target
                
                if model_key not in route_models:
                    continue
                
                feature_cols = route_feature_cols[model_key]
                model = route_models[model_key]
                
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
            
            # Lưu predictions của ngày này để dùng cho ngày tiếp theo
            if len(all_predictions) > 0:
                previous_day_predictions[forecast_date] = all_predictions.copy()
                
                # Cập nhật available_data với tất cả predictions để tính features cho ngày sau
                for idx, hour in enumerate(hours):
                    dt = pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=hour)
                    new_row = forecast_df.iloc[idx].copy()
                    
                    # Cập nhật tất cả các attributes đã predict
                    for attr_name, pred_values in all_predictions.items():
                        if idx < len(pred_values):
                            new_row[attr_name] = float(pred_values[idx])
                    
                    # Thêm vào available_data để tính features
                    new_row_df = pd.DataFrame([new_row])
                    available_data = pd.concat([available_data, new_row_df], ignore_index=True)
                
                available_data = available_data.sort_values('datetime').reset_index(drop=True)
                # Tạo lại features sau khi thêm dữ liệu mới
                available_data = create_features_for_prediction(available_data, 'Temp')
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"❌ Error in api_forecast: {error_msg}")
        print(f"Traceback: {error_traceback}")
        return jsonify({'error': error_msg, 'traceback': error_traceback}), 500

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin/api/tables', methods=['GET'])
def admin_get_tables():
    try:
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/api/stats', methods=['GET'])
def admin_get_stats():
    try:
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        stats['weather_data_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT city) FROM weather_data")
        stats['cities_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM weather_data")
        date_range = cursor.fetchone()
        stats['date_range'] = {
            'min': date_range[0],
            'max': date_range[1]
        }
        
        cursor.execute("SELECT COUNT(*) FROM accuracy_results")
        stats['accuracy_results_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        stats['tables_count'] = len(cursor.fetchall())
        
        conn.close()
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/api/table/<table_name>', methods=['GET'])
def admin_get_table_data(table_name):
    try:
        from database import get_db_connection
        import sqlite3
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 100, type=int)
        search = request.args.get('search', '', type=str)
        city = request.args.get('city', '', type=str)
        date_from = request.args.get('date_from', '', type=str)
        date_to = request.args.get('date_to', '', type=str)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if table_name not in ['weather_data', 'accuracy_results']:
            conn.close()
            return jsonify({'error': 'Invalid table name'}), 400
        
        query = f"SELECT * FROM {table_name}"
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        conditions = []
        params = []
        
        if table_name == 'weather_data':
            if search:
                conditions.append("(city LIKE ? OR date LIKE ? OR Time LIKE ?)")
                search_param = f"%{search}%"
                params.extend([search_param, search_param, search_param])
            if city:
                conditions.append("city = ?")
                params.append(city)
            if date_from:
                conditions.append("date >= ?")
                params.append(date_from)
            if date_to:
                conditions.append("date <= ?")
                params.append(date_to)
        elif table_name == 'accuracy_results':
            if search:
                conditions.append("(city LIKE ? OR calculated_at LIKE ?)")
                search_param = f"%{search}%"
                params.extend([search_param, search_param])
            if city:
                conditions.append("city = ?")
                params.append(city)
        
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause
        
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        offset = (page - 1) * per_page
        query += f" ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                if value is None:
                    row_dict[col] = None
                elif isinstance(value, (int, float)):
                    row_dict[col] = value
                else:
                    row_dict[col] = str(value)
            data.append(row_dict)
        
        conn.close()
        
        return jsonify({
            'data': data,
            'columns': columns,
            'total': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_count + per_page - 1) // per_page
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/admin/api/cities', methods=['GET'])
def admin_get_cities():
    try:
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT city FROM weather_data ORDER BY city")
        cities = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({'cities': cities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/climate')
def climate():
    return render_template('climate.html')

@app.route('/api/statistics/descriptive', methods=['POST'])
def api_statistics_descriptive():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year_from = data.get('year_from', 2017)
        year_to = data.get('year_to', 2025)
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] >= year_from) & 
            (df['year'] <= year_to)
        ].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found for {city} from {year_from} to {year_to}'}), 400
        
        numeric_cols = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
        result = {}
        
        for col in numeric_cols:
            if col in city_df.columns:
                values = city_df[col].dropna()
                if len(values) > 0:
                    result[col] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75)),
                        'count': int(len(values))
                    }
        
        return jsonify({
            'city': city,
            'year_from': year_from,
            'year_to': year_to,
            'statistics': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics/distribution', methods=['POST'])
def api_statistics_distribution():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        metric = data.get('metric', 'Temp')
        year_from = data.get('year_from', 2017)
        year_to = data.get('year_to', 2025)
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] >= year_from) & 
            (df['year'] <= year_to)
        ].copy()
        
        if len(city_df) == 0 or metric not in city_df.columns:
            return jsonify({'error': f'No data found'}), 400
        
        values = city_df[metric].dropna()
        if len(values) == 0:
            return jsonify({'error': 'No valid values'}), 400
        
        min_val = float(values.min())
        max_val = float(values.max())
        bins = 20
        bin_width = (max_val - min_val) / bins
        
        histogram = [0] * bins
        for val in values:
            bin_idx = min(int((val - min_val) / bin_width), bins - 1)
            histogram[bin_idx] += 1
        
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(bins)]
        
        return jsonify({
            'city': city,
            'metric': metric,
            'year_from': year_from,
            'year_to': year_to,
            'histogram': {
                'bins': histogram,
                'bin_centers': [float(x) for x in bin_centers],
                'bin_edges': [float(x) for x in bin_edges]
            },
            'min': float(min_val),
            'max': float(max_val)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics/correlation', methods=['POST'])
def api_statistics_correlation():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        year_from = data.get('year_from', 2017)
        year_to = data.get('year_to', 2025)
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] >= year_from) & 
            (df['year'] <= year_to)
        ].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found'}), 400
        
        numeric_cols = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
        available_cols = [col for col in numeric_cols if col in city_df.columns]
        
        correlation_matrix = city_df[available_cols].corr()
        
        result = {}
        for i, col1 in enumerate(available_cols):
            result[col1] = {}
            for col2 in available_cols:
                result[col1][col2] = float(correlation_matrix.loc[col1, col2])
        
        return jsonify({
            'city': city,
            'year_from': year_from,
            'year_to': year_to,
            'correlation': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics/trend', methods=['POST'])
def api_statistics_trend():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        metric = data.get('metric', 'Temp')
        year_from = data.get('year_from', 2017)
        year_to = data.get('year_to', 2025)
        
        df = load_data()
        city_df = df[
            (df['city'] == city) & 
            (df['year'] >= year_from) & 
            (df['year'] <= year_to)
        ].copy()
        
        if len(city_df) == 0 or metric not in city_df.columns:
            return jsonify({'error': f'No data found'}), 400
        
        yearly_data = []
        monthly_data = []
        seasonal_data = []
        
        for year in sorted(city_df['year'].unique()):
            year_df = city_df[city_df['year'] == year]
            yearly_data.append({
                'year': int(year),
                'mean': float(year_df[metric].mean()),
                'min': float(year_df[metric].min()),
                'max': float(year_df[metric].max())
            })
        
        for month in range(1, 13):
            month_df = city_df[city_df['month'] == month]
            if len(month_df) > 0:
                monthly_data.append({
                    'month': month,
                    'mean': float(month_df[metric].mean()),
                    'min': float(month_df[metric].min()),
                    'max': float(month_df[metric].max())
                })
        
        season_names = {0: 'Đông', 1: 'Xuân', 2: 'Hè', 3: 'Thu'}
        for season in range(4):
            season_df = city_df[city_df['season'] == season]
            if len(season_df) > 0:
                seasonal_data.append({
                    'season': season,
                    'season_name': season_names[season],
                    'mean': float(season_df[metric].mean()),
                    'min': float(season_df[metric].min()),
                    'max': float(season_df[metric].max())
                })
        
        return jsonify({
            'city': city,
            'metric': metric,
            'year_from': year_from,
            'year_to': year_to,
            'yearly': yearly_data,
            'monthly': monthly_data,
            'seasonal': seasonal_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate/trend', methods=['POST'])
def api_climate_trend():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        metric = data.get('metric', 'Temp')
        
        df = load_data()
        city_df = df[df['city'] == city].copy()
        
        if len(city_df) == 0 or metric not in city_df.columns:
            return jsonify({'error': f'No data found'}), 400
        
        yearly_avg = []
        for year in sorted(city_df['year'].unique()):
            year_df = city_df[city_df['year'] == year]
            yearly_avg.append({
                'year': int(year),
                'value': float(year_df[metric].mean())
            })
        
        if len(yearly_avg) >= 2:
            first_year = yearly_avg[0]['value']
            last_year = yearly_avg[-1]['value']
            total_change = last_year - first_year
            years_span = yearly_avg[-1]['year'] - yearly_avg[0]['year']
            annual_change = total_change / years_span if years_span > 0 else 0
        else:
            total_change = 0
            annual_change = 0
        
        return jsonify({
            'city': city,
            'metric': metric,
            'yearly_average': yearly_avg,
            'total_change': float(total_change),
            'annual_change': float(annual_change),
            'first_year': int(yearly_avg[0]['year']) if yearly_avg else None,
            'last_year': int(yearly_avg[-1]['year']) if yearly_avg else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate/seasonal', methods=['POST'])
def api_climate_seasonal():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        metric = data.get('metric', 'Temp')
        
        df = load_data()
        city_df = df[df['city'] == city].copy()
        
        if len(city_df) == 0 or metric not in city_df.columns:
            return jsonify({'error': f'No data found'}), 400
        
        season_names = {0: 'Đông', 1: 'Xuân', 2: 'Hè', 3: 'Thu'}
        years = sorted(city_df['year'].unique())
        
        result = {}
        for season in range(4):
            season_name = season_names[season]
            result[season_name] = []
            
            for year in years:
                year_season_df = city_df[(city_df['year'] == year) & (city_df['season'] == season)]
                if len(year_season_df) > 0:
                    result[season_name].append({
                        'year': int(year),
                        'mean': float(year_season_df[metric].mean()),
                        'min': float(year_season_df[metric].min()),
                        'max': float(year_season_df[metric].max())
                    })
        
        return jsonify({
            'city': city,
            'metric': metric,
            'seasonal_data': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate/changes', methods=['POST'])
def api_climate_changes():
    try:
        data = request.json
        city = data.get('city', 'vinh')
        period1_start = data.get('period1_start', 2017)
        period1_end = data.get('period1_end', 2020)
        period2_start = data.get('period2_start', 2021)
        period2_end = data.get('period2_end', 2025)
        
        df = load_data()
        city_df = df[df['city'] == city].copy()
        
        if len(city_df) == 0:
            return jsonify({'error': f'No data found'}), 400
        
        numeric_cols = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
        result = {}
        
        for col in numeric_cols:
            if col not in city_df.columns:
                continue
            
            period1_df = city_df[
                (city_df['year'] >= period1_start) & 
                (city_df['year'] <= period1_end)
            ]
            period2_df = city_df[
                (city_df['year'] >= period2_start) & 
                (city_df['year'] <= period2_end)
            ]
            
            if len(period1_df) > 0 and len(period2_df) > 0:
                period1_mean = float(period1_df[col].mean())
                period2_mean = float(period2_df[col].mean())
                change = period2_mean - period1_mean
                change_percent = (change / period1_mean * 100) if period1_mean != 0 else 0
                
                result[col] = {
                    'period1': {
                        'start': period1_start,
                        'end': period1_end,
                        'mean': period1_mean
                    },
                    'period2': {
                        'start': period2_start,
                        'end': period2_end,
                        'mean': period2_mean
                    },
                    'change': float(change),
                    'change_percent': float(change_percent)
                }
        
        return jsonify({
            'city': city,
            'comparison': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

