#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tự động tạo forecast hàng ngày và lưu vào database
Chạy script này mỗi ngày để cập nhật dữ liệu dự báo vào bảng system_forecasts
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
import pickle
import warnings
import json

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*xgboost.*')

# Import các hàm từ app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import các hàm cần thiết từ app.py
from app import (
    load_data, 
    create_features_for_prediction, 
    apply_cold_air_adjustment,
    get_models_for_route,
    get_cold_air_setting
)

target_vars = ['Temp_numeric', 'Rain_numeric', 'Cloud_numeric', 
               'Pressure_numeric', 'Wind_numeric', 'Gust_numeric']

def generate_forecast_for_city(city, today):
    """Tạo forecast cho một thành phố và lưu vào database"""
    try:
        print(f"\n[{city}] Đang tạo forecast cho ngày {today.strftime('%Y-%m-%d')}...")
        
        # Load models
        route_models, route_feature_cols = get_models_for_route(use_improved=False)
        if not route_models:
            print(f"  ❌ Không tìm thấy models!")
            return False
        
        # Load dữ liệu
        df = load_data()
        city_df = df[df['city'] == city].copy().sort_values('datetime')
        
        if len(city_df) == 0:
            print(f"  ❌ Không có dữ liệu cho {city}")
            return False
        
        last_date = city_df['datetime'].dt.date.max()
        
        # Tạo forecast cho ngày hôm nay
        forecast_date = today
        
        # Lấy dữ liệu để tạo features
        start_date_for_features = forecast_date - pd.Timedelta(days=2)
        available_data = city_df[
            (city_df['datetime'].dt.date >= start_date_for_features) & 
            (city_df['datetime'].dt.date <= last_date)
        ].copy().sort_values('datetime')
        
        if len(available_data) == 0:
            print(f"  ❌ Không có dữ liệu đủ để tạo forecast")
            return False
        
        available_data = create_features_for_prediction(available_data, 'Temp')
        
        # Tạo forecast cho 8 giờ trong ngày
        hours = [0, 3, 6, 9, 12, 15, 18, 21]
        forecast_data_list = []
        
        for hour in hours:
            dt = pd.Timestamp.combine(forecast_date, pd.Timestamp.min.time()) + pd.Timedelta(hours=hour)
            
            # Lấy dữ liệu gần nhất
            real_data = city_df[city_df['datetime'].dt.date < forecast_date].copy()
            latest_row = pd.Series()
            
            if len(real_data) > 0:
                same_hour_real = real_data[(real_data['datetime'].dt.hour == hour)]
                if len(same_hour_real) > 0:
                    numeric_cols = same_hour_real.select_dtypes(include=[np.number]).columns
                    latest_row = same_hour_real.tail(7)[numeric_cols].mean()
                    last_row = same_hour_real.tail(1).iloc[0]
                    for col in same_hour_real.columns:
                        if col not in numeric_cols:
                            latest_row[col] = last_row[col]
                else:
                    numeric_cols = real_data.select_dtypes(include=[np.number]).columns
                    latest_row = real_data.tail(24)[numeric_cols].mean()
                    last_row = real_data.tail(1).iloc[0]
                    for col in real_data.columns:
                        if col not in numeric_cols:
                            latest_row[col] = last_row[col]
            
            # Tạo row cho forecast
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
            
            # Season
            if forecast_date.month in [12, 1, 2]:
                row['season'] = 0
            elif forecast_date.month in [3, 4, 5]:
                row['season'] = 1
            elif forecast_date.month in [6, 7, 8]:
                row['season'] = 2
            else:
                row['season'] = 3
            
            # Features
            row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            row['month_sin'] = np.sin(2 * np.pi * forecast_date.month / 12)
            row['month_cos'] = np.cos(2 * np.pi * forecast_date.month / 12)
            row['day_of_year_sin'] = np.sin(2 * np.pi * row['day_of_year'] / 365)
            row['day_of_year_cos'] = np.cos(2 * np.pi * row['day_of_year'] / 365)
            
            # Giá trị mặc định
            for col in ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']:
                if col in latest_row.index and pd.notna(latest_row[col]):
                    row[col] = float(latest_row[col])
                else:
                    defaults = {'Temp': 25.0, 'Rain': 0.0, 'Cloud': 50.0, 
                               'Pressure': 1013.0, 'Wind': 10.0, 'Gust': 15.0, 'Dir': 0}
                    row[col] = defaults.get(col, 0.0)
            
            # Thêm các features từ available_data
            for col in available_data.columns:
                if col not in row and col.startswith(('Temp_', 'Pressure_', 'Wind_', 'Cloud_')):
                    if col in latest_row.index and pd.notna(latest_row[col]):
                        row[col] = latest_row[col]
                    else:
                        row[col] = 0.0
            
            forecast_data_list.append(row)
        
        forecast_df = pd.DataFrame(forecast_data_list)
        
        # Tạo features cho forecast_df
        forecast_df = create_features_for_prediction(forecast_df, 'Temp')
        
        # Dự đoán
        all_predictions = {}
        for target in target_vars:
            if target == 'Temp_numeric' and city == 'ho-chi-minh-city' and 'Temp_numeric_hcm' in route_models:
                model_key = 'Temp_numeric_hcm'
            else:
                model_key = target
            
            if model_key not in route_models:
                continue
            
            feature_cols = route_feature_cols[model_key]
            model = route_models[model_key]
            
            # Map features
            feature_mapping = {}
            for col in feature_cols:
                if col.endswith('_numeric') and col.replace('_numeric', '') in forecast_df.columns:
                    feature_mapping[col] = col.replace('_numeric', '')
                elif col in forecast_df.columns:
                    feature_mapping[col] = col
            
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
            
            if target == 'Temp_numeric':
                predictions = apply_cold_air_adjustment(predictions, city, forecast_df['hour'].values if 'hour' in forecast_df.columns else None)
            
            target_name = target.replace('_numeric', '')
            all_predictions[target_name] = predictions
        
        # Tính toán giá trị trung bình, min, max cho ngày
        forecast_avg = {}
        forecast_min = {}
        forecast_max = {}
        
        for attr_name, pred_values in all_predictions.items():
            forecast_avg[attr_name] = float(np.mean(pred_values))
            forecast_min[attr_name] = float(np.min(pred_values))
            forecast_max[attr_name] = float(np.max(pred_values))
        
        # Lưu vào database
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO system_forecasts 
            (city, date, Temp, Temp_min, Temp_max, Pressure, Wind, Rain, Cloud, Gust)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            city,
            today.strftime('%Y-%m-%d'),
            forecast_avg.get('Temp'),
            forecast_min.get('Temp'),
            forecast_max.get('Temp'),
            forecast_avg.get('Pressure'),
            forecast_avg.get('Wind'),
            forecast_avg.get('Rain'),
            forecast_avg.get('Cloud'),
            forecast_avg.get('Gust')
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Đã tạo và lưu forecast cho {city} - {today.strftime('%Y-%m-%d')}")
        print(f"     Temp: {forecast_avg.get('Temp', 0):.1f}°C (min: {forecast_min.get('Temp', 0):.1f}, max: {forecast_max.get('Temp', 0):.1f})")
        return True
        
    except Exception as e:
        print(f"  ❌ Lỗi khi tạo forecast cho {city}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hàm chính"""
    print("="*70)
    print("TỰ ĐỘNG TẠO FORECAST HÀNG NGÀY")
    print("="*70)
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models
    print("\nĐang load models...")
    try:
        # Import và load models từ app.py
        from app import load_models
        if not load_models():
            print("❌ Không thể load models!")
            return False
        print("✅ Đã load models thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return False
    
    # Ngày hôm nay
    today = date.today()
    
    # Danh sách thành phố
    cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    
    # Tạo forecast cho từng thành phố
    success_count = 0
    for city in cities:
        if generate_forecast_for_city(city, today):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"HOÀN TẤT: {success_count}/{len(cities)} thành phố đã được tạo forecast")
    print("="*70)
    
    return success_count == len(cities)

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã dừng bởi người dùng")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
