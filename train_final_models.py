# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("="*70)
print("TRAINING CÁC MODEL CUỐI CÙNG VỚI TOÀN BỘ DỮ LIỆU (CÁCH MỚI)")
print("="*70)

def load_and_preprocess_data(file_path=None):
    print("\n[1] Đang tải và tiền xử lý dữ liệu...")
    
    try:
        from database import load_data_from_db
        import os
        
        if os.path.exists('weather.db'):
            df = load_data_from_db()
            if len(df) > 0:
                if len(df) >= 1000:
                    print(f"  ✓ Đã tải {len(df)} bản ghi từ database")
                    print(f"  Khoảng thời gian: {df['datetime'].min()} đến {df['datetime'].max()}")
                    return df
                else:
                    print(f"  ⚠ Database chỉ có {len(df)} bản ghi (không đủ để train)")
                    print("  ⚠ Chuyển sang CSV để lấy toàn bộ dữ liệu...")
        
        print("  ⚠ Không tìm thấy database hoặc database trống. Chuyển sang CSV...")
    except Exception as e:
        print(f"  ⚠ Lỗi khi tải từ database: {e}. Chuyển sang CSV...")
    
    if file_path is None:
        file_path = 'weather_all_cities.csv'
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
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
    
    print(f"  ✓ Đã tải {len(df)} bản ghi từ CSV")
    print(f"  Khoảng thời gian: {df['datetime'].min()} đến {df['datetime'].max()}")
    
    return df

def create_advanced_features_for_all(df):
    print(f"\n[2] Đang tạo các feature nâng cao cho tất cả các biến...")
    
    df = df.copy()
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_data = df[city_mask].copy().sort_values('datetime').reset_index(drop=True)
        
        for lag in [12, 24]:
            df.loc[city_mask, f'Temp_lag_{lag}'] = city_data['Temp'].shift(lag).values
        
        for window in [12, 24]:
            df.loc[city_mask, f'Temp_rolling_mean_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Temp_rolling_std_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[city_mask, f'Temp_rolling_max_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).max().values
            df.loc[city_mask, f'Temp_rolling_min_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).min().values
        
        for var in ['Pressure', 'Wind', 'Cloud', 'Rain', 'Gust']:
            for lag in [1, 3, 6, 12, 24]:
                df.loc[city_mask, f'{var}_lag_{lag}'] = city_data[var].shift(lag).values
        
        for var in ['Pressure', 'Wind', 'Cloud', 'Rain', 'Gust']:
            for window in [3, 6, 12, 24]:
                df.loc[city_mask, f'{var}_rolling_mean_{window}'] = city_data[var].shift(1).rolling(window=window, min_periods=1).mean().values
                df.loc[city_mask, f'{var}_rolling_max_{window}'] = city_data[var].shift(1).rolling(window=window, min_periods=1).max().values
        
        has_rain = (city_data['Rain'] > 0).astype(int)
        df.loc[city_mask, 'has_rain'] = has_rain.values
        df.loc[city_mask, 'has_rain_lag_1'] = has_rain.shift(1).fillna(0).astype(int).values
        df.loc[city_mask, 'has_rain_lag_3'] = has_rain.shift(3).fillna(0).astype(int).values
        df.loc[city_mask, 'has_rain_lag_6'] = has_rain.shift(6).fillna(0).astype(int).values
        
        city_data['hour'] = city_data['datetime'].dt.hour
        df.loc[city_mask, 'Temp_same_hour_1d_ago'] = city_data.groupby('hour')['Temp'].shift(8).values
        df.loc[city_mask, 'Temp_same_hour_7d_ago'] = city_data.groupby('hour')['Temp'].shift(7*8).values
        df.loc[city_mask, 'Temp_same_hour_avg_7d'] = city_data.groupby('hour')['Temp'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        df.loc[city_mask, 'Rain_same_hour_1d_ago'] = city_data.groupby('hour')['Rain'].shift(8).values
        df.loc[city_mask, 'Rain_same_hour_7d_ago'] = city_data.groupby('hour')['Rain'].shift(7*8).values
        df.loc[city_mask, 'Rain_same_hour_avg_7d'] = city_data.groupby('hour')['Rain'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        df.loc[city_mask, 'Cloud_same_hour_1d_ago'] = city_data.groupby('hour')['Cloud'].shift(8).values
        df.loc[city_mask, 'Cloud_same_hour_7d_ago'] = city_data.groupby('hour')['Cloud'].shift(7*8).values
        df.loc[city_mask, 'Cloud_same_hour_avg_7d'] = city_data.groupby('hour')['Cloud'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
    
    df['hour_month_interaction'] = df['hour'] * df['month']
    df['pressure_wind_interaction'] = df['Pressure'] * df['Wind'] / 100
    df['cloud_hour_interaction'] = df['Cloud'] * df['hour'] / 100
    df['cloud_pressure_interaction'] = df['Cloud'] * df['Pressure'] / 1000
    df['rain_cloud_interaction'] = df['Rain'] * df['Cloud'] / 100
    df['temp_pressure_interaction'] = df['Temp'] * df['Pressure'] / 100
    
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    print(f"  ✓ Đã tạo các feature nâng cao cho tất cả các biến")
    return df

def train_final_model(df, target='Temp'):
    print(f"\n[3] Đang train model cuối cùng cho {target}...")
    
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    if target == 'Gust':
        base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season']
    
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    
    weather_features = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
    weather_features = [f for f in weather_features if f != target]
    
    temp_lag_features = ['Temp_lag_12', 'Temp_lag_24']
    temp_rolling_features = ['Temp_rolling_mean_12', 'Temp_rolling_std_12', 
                             'Temp_rolling_max_12', 'Temp_rolling_min_12',
                             'Temp_rolling_mean_24', 'Temp_rolling_std_24',
                             'Temp_rolling_max_24', 'Temp_rolling_min_24']
    
    if target == 'Temp':
        other_vars_lag_rolling = []
        for var in ['Pressure', 'Wind', 'Cloud']:
            for lag in [1, 3, 6, 12]:
                other_vars_lag_rolling.append(f'{var}_lag_{lag}')
            for window in [3, 6, 12, 24]:
                other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
    else:
        other_vars_lag_rolling = []
        for var in ['Pressure', 'Wind', 'Cloud', 'Rain']:
            if var != target:
                if target == 'Pressure':
                    if var == 'Wind':
                        for lag in [1, 24]:
                            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
                    elif var == 'Cloud':
                        for lag in [1, 3]:
                            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
                    elif var == 'Rain':
                        for lag in [1, 24]:
                            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
                    else:
                        for lag in [1, 3, 6, 12, 24]:
                            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
                else:
                    for lag in [1, 3, 6, 12, 24]:
                        other_vars_lag_rolling.append(f'{var}_lag_{lag}')
                
                for window in [3, 6, 12, 24]:
                    other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
                    other_vars_lag_rolling.append(f'{var}_rolling_max_{window}')
            else:
                for lag in [1, 3, 6, 12, 24]:
                    other_vars_lag_rolling.append(f'{target}_lag_{lag}')
                for window in [3, 6, 12, 24]:
                    other_vars_lag_rolling.append(f'{target}_rolling_mean_{window}')
                    other_vars_lag_rolling.append(f'{target}_rolling_max_{window}')
    
    if target == 'Temp':
        context_features = ['Temp_same_hour_1d_ago', 'Temp_same_hour_7d_ago', 'Temp_same_hour_avg_7d']
    else:
        context_features = ['Temp_same_hour_1d_ago', 'Temp_same_hour_7d_ago', 'Temp_same_hour_avg_7d']
        
        if target != 'Rain':
            if target == 'Pressure':
                context_features.extend(['Rain_same_hour_1d_ago', 'Rain_same_hour_avg_7d'])
            else:
                context_features.extend(['Rain_same_hour_1d_ago', 'Rain_same_hour_7d_ago', 'Rain_same_hour_avg_7d'])
        
        if target != 'Cloud':
            context_features.extend(['Cloud_same_hour_1d_ago', 'Cloud_same_hour_7d_ago', 'Cloud_same_hour_avg_7d'])
        
        if target == 'Rain':
            context_features.extend(['Rain_same_hour_1d_ago', 'Rain_same_hour_7d_ago', 'Rain_same_hour_avg_7d'])
        
        if target == 'Cloud':
            context_features.extend(['Cloud_same_hour_1d_ago', 'Cloud_same_hour_7d_ago', 'Cloud_same_hour_avg_7d'])
    
    if target == 'Temp':
        rain_binary_features = []
    elif target == 'Pressure':
        rain_binary_features = ['has_rain_lag_3', 'has_rain_lag_6']
    elif target == 'Rain':
        rain_binary_features = ['has_rain_lag_1']
    elif target == 'Gust':
        rain_binary_features = ['has_rain_lag_3']
    elif target == 'Wind':
        rain_binary_features = ['has_rain_lag_1', 'has_rain_lag_6']
    else:
        rain_binary_features = ['has_rain_lag_1', 'has_rain_lag_3', 'has_rain_lag_6']
    
    if target == 'Temp':
        interaction_features = [
            'hour_month_interaction',
            'pressure_wind_interaction',
            'cloud_hour_interaction'
        ]
    else:
        all_interaction_features = {
            'hour_month_interaction': ['hour', 'month'],
            'pressure_wind_interaction': ['Pressure', 'Wind'],
            'cloud_hour_interaction': ['Cloud', 'hour'],
            'cloud_pressure_interaction': ['Cloud', 'Pressure'],
            'rain_cloud_interaction': ['Rain', 'Cloud'],
            'temp_pressure_interaction': ['Temp', 'Pressure']
        }
        
        interaction_features = []
        for inter_name, vars_in_inter in all_interaction_features.items():
            if target not in vars_in_inter:
                interaction_features.append(inter_name)
    
    if target == 'Temp':
        city_features = [col for col in df.columns if col.startswith('city_')]
    else:
        city_features = []
    
    features_to_remove = []
    features_to_remove.append('Rain_same_hour_7d_ago')
    
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   rain_binary_features + interaction_features + city_features)
    
    all_features = [f for f in all_features if f not in features_to_remove]
    
    feature_cols = [col for col in all_features if col in df.columns]
    
    print(f"  Using {len(feature_cols)} features")
    
    df_clean = df.dropna(subset=[target]).copy()
    
    if 'Temp_lag_24' in df_clean.columns:
        df_clean = df_clean[df_clean['Temp_lag_24'].notna()].copy()
    
    if len(df_clean) == 0:
        print(f"  ⚠ Không có dữ liệu để train {target}")
        return None, []
    
    X = df_clean[feature_cols].fillna(0).copy()
    y = df_clean[target].copy()
    
    for col in X.columns:
        if isinstance(X[col].iloc[0] if len(X) > 0 else None, pd.DataFrame):
            X[col] = X[col].apply(lambda x: x.iloc[0, 0] if isinstance(x, pd.DataFrame) else x)
    
    print(f"  Đang train trên {len(X)} mẫu")
    
    if target == 'Rain':
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.3,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        )
    elif target == 'Cloud':
        model = XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=4,
            gamma=0.25,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    
    print("  Đang train model...")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    def calculate_mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    mape = calculate_mape(y, y_pred)
    
    print(f"\n  Kết quả (trên toàn bộ dữ liệu):")
    print(f"    R²: {r2_score(y, y_pred):.4f}")
    print(f"    MAE: {mean_absolute_error(y, y_pred):.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    
    return model, feature_cols

def train_final_hcm_model(df):
    print(f"\n[4] Đang train model riêng cho HCM...")
    
    hcm_df = df[df['city'] == 'ho-chi-minh-city'].copy()
    print(f"  Dữ liệu HCM: {len(hcm_df)} bản ghi")
    
    print(f"\n[4] Đang train model HCM cuối cùng cho Temp...")
    
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    weather_features = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
    temp_lag_features = ['Temp_lag_12', 'Temp_lag_24']
    temp_rolling_features = ['Temp_rolling_mean_12', 'Temp_rolling_std_12', 
                             'Temp_rolling_max_12', 'Temp_rolling_min_12',
                             'Temp_rolling_mean_24', 'Temp_rolling_std_24',
                             'Temp_rolling_max_24', 'Temp_rolling_min_24']
    
    other_vars_lag_rolling = []
    for var in ['Pressure', 'Wind', 'Cloud']:
        for lag in [1, 3, 6, 12]:
            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
        for window in [3, 6, 12, 24]:
            other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
    
    context_features = ['Temp_same_hour_1d_ago', 'Temp_same_hour_7d_ago', 'Temp_same_hour_avg_7d']
    interaction_features = ['hour_month_interaction', 'pressure_wind_interaction', 'cloud_hour_interaction']
    city_features = [col for col in hcm_df.columns if col.startswith('city_')]
    
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   interaction_features + city_features)
    
    feature_cols = [col for col in all_features if col in hcm_df.columns]
    
    print(f"  Sử dụng {len(feature_cols)} features")
    
    hcm_df_clean = hcm_df.dropna(subset=['Temp']).copy()
    if 'Temp_lag_24' in hcm_df_clean.columns:
        hcm_df_clean = hcm_df_clean[hcm_df_clean['Temp_lag_24'].notna()].copy()
    
    if len(hcm_df_clean) == 0:
        print(f"  ⚠ Không có dữ liệu để train model HCM")
        return None, []
    
    X = hcm_df_clean[feature_cols].fillna(0).copy()
    y = hcm_df_clean['Temp'].copy()
    
    print(f"  Đang train trên {len(X)} mẫu")
    
    model = XGBRegressor(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Đang train model...")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    def calculate_mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    mape = calculate_mape(y, y_pred)
    
    print(f"\n  Kết quả (trên toàn bộ dữ liệu):")
    print(f"    R²: {r2_score(y, y_pred):.4f}")
    print(f"    MAE: {mean_absolute_error(y, y_pred):.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    
    return model, feature_cols

def save_final_models(models_dict, feature_cols_dict, filename='weather_models_final.pkl'):
    print(f"\n[5] Đang lưu các model cuối cùng vào {filename}...")
    
    models_data = {
        'models': models_dict,
        'feature_cols': feature_cols_dict
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(models_data, f)
    
    print(f"  ✓ Đã lưu {len(models_dict)} models vào {filename}")
    print(f"  Models: {list(models_dict.keys())}")

if __name__ == '__main__':
    df = load_and_preprocess_data()
    
    df = create_advanced_features_for_all(df)
    
    target_vars = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    models_dict = {}
    feature_cols_dict = {}
    
    for target in target_vars:
        model, feature_cols = train_final_model(df, target)
        if model is not None:
            models_dict[f'{target}_numeric'] = model
            feature_cols_dict[f'{target}_numeric'] = feature_cols
    
    hcm_model, hcm_feature_cols = train_final_hcm_model(df)
    if hcm_model is not None:
        models_dict['Temp_numeric_hcm'] = hcm_model
        feature_cols_dict['Temp_numeric_hcm'] = hcm_feature_cols
    
    save_final_models(models_dict, feature_cols_dict, 'weather_models_final.pkl')
    
    print("\n" + "="*70)
    print("HOÀN TẤT TRAINING CÁC MODEL CUỐI CÙNG!")
    print("="*70)
    print("\nCác model đã được train trên TẤT CẢ dữ liệu có sẵn (2017-2026-01-02)")
    print("Sử dụng CÁCH MỚI: Mỗi model có lag/rolling features của chính nó")
    print("Sẵn sàng để dự báo các ngày trong tương lai!")
