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
print("TRAINING TẤT CẢ CÁC MODEL CẢI TIẾN VỚI FEATURES NÂNG CAO")
print("="*70)

def load_and_preprocess_data(file_path=None):
    print("\n[1] Đang tải và tiền xử lý dữ liệu...")
    
    try:
        from database import load_data_from_db
        import os
        
        if os.path.exists('weather.db'):
            df = load_data_from_db()
            if len(df) > 0:
                print(f"  ✓ Đã tải {len(df)} bản ghi từ database")
                return df
        
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
    
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    print(f"  ✓ Đã tải {len(df)} bản ghi từ CSV")
    
    return df

def split_data_by_time(df):
    print("\n[2] Đang chia dữ liệu theo thời gian...")
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    val_start = '2021-01-01'
    
    train_df = df[df['datetime'] <= train_end].copy()
    test_df = df[df['datetime'] >= test_start].copy()
    val_df = train_df[train_df['datetime'] >= val_start].copy()
    train_df_only = train_df[train_df['datetime'] < val_start].copy()
    
    print(f"  - Training: {len(train_df_only)} bản ghi")
    print(f"  - Validation: {len(val_df)} bản ghi")
    print(f"  - Test: {len(test_df)} bản ghi")
    
    return train_df_only, val_df, test_df

def create_advanced_features_for_temp(train_df, val_df, test_df):
    print(f"\n[PHẦN 1] Đang tạo các feature nâng cao cho model Temp...")
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    target_col = 'Temp'
    
    for df_split, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df_split = df_split.sort_values(['city', 'datetime']).reset_index(drop=True)
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            
            for lag in [12, 24]:
                df_split.loc[city_mask, f'{target_col}_lag_{lag}'] = city_data[target_col].shift(lag).values
            
            for window in [12, 24]:
                df_split.loc[city_mask, f'{target_col}_rolling_mean_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split.loc[city_mask, f'{target_col}_rolling_std_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
                df_split.loc[city_mask, f'{target_col}_rolling_max_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).max().values
                df_split.loc[city_mask, f'{target_col}_rolling_min_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).min().values
            
            for lag in [1, 3, 6, 12]:
                df_split.loc[city_mask, f'Pressure_lag_{lag}'] = city_data['Pressure'].shift(lag).values
                df_split.loc[city_mask, f'Wind_lag_{lag}'] = city_data['Wind'].shift(lag).values
                df_split.loc[city_mask, f'Cloud_lag_{lag}'] = city_data['Cloud'].shift(lag).values
            
            for window in [3, 6, 12, 24]:
                df_split.loc[city_mask, f'Pressure_rolling_mean_{window}'] = city_data['Pressure'].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split.loc[city_mask, f'Wind_rolling_mean_{window}'] = city_data['Wind'].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split.loc[city_mask, f'Cloud_rolling_mean_{window}'] = city_data['Cloud'].shift(1).rolling(window=window, min_periods=1).mean().values
            
            city_data['hour'] = city_data['datetime'].dt.hour
            df_split.loc[city_mask, f'{target_col}_same_hour_1d_ago'] = city_data.groupby('hour')[target_col].shift(8).values
            df_split.loc[city_mask, f'{target_col}_same_hour_7d_ago'] = city_data.groupby('hour')[target_col].shift(7*8).values
                df_split.loc[city_mask, f'{target_col}_same_hour_avg_7d'] = city_data.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        df_split['hour_month_interaction'] = df_split['hour'] * df_split['month']
        df_split['pressure_wind_interaction'] = df_split['Pressure'] * df_split['Wind'] / 100
        df_split['cloud_hour_interaction'] = df_split['Cloud'] * df_split['hour'] / 100
        
        city_dummies = pd.get_dummies(df_split['city'], prefix='city')
        all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
        for city in all_cities:
            col_name = f'city_{city}'
            if col_name not in city_dummies.columns:
                city_dummies[col_name] = 0
        
        for col in city_dummies.columns:
            if col not in df_split.columns:
                df_split[col] = city_dummies[col]
        
        if split_name == 'train':
            train_df = df_split
        elif split_name == 'val':
            val_df = df_split
        else:
            test_df = df_split
    
    print(f"  ✓ Đã tạo các feature nâng cao cho model Temp")
    return train_df, val_df, test_df

def train_improved_temp_model(train_df, val_df, test_df):
    print("\n[PHẦN 1] Đang train model nhiệt độ cải tiến...")
    
    target = 'Temp'
    
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    
    weather_features = ['Rain', 'Cloud', 'Pressure', 
                       'Wind', 'Gust', 'Dir']
    
    temp_lag_features = [f'{target}_lag_{lag}' for lag in [12, 24]]
    
    temp_rolling_features = []
    for window in [12, 24]:
        temp_rolling_features.extend([
            f'{target}_rolling_mean_{window}',
            f'{target}_rolling_std_{window}',
            f'{target}_rolling_max_{window}',
            f'{target}_rolling_min_{window}'
        ])
    
    other_vars_lag_rolling = []
    for var in ['Pressure', 'Wind', 'Cloud']:
        for lag in [1, 3, 6, 12]:
            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
        for window in [3, 6, 12, 24]:
            other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
    
    context_features = [
        f'{target}_same_hour_1d_ago',
        f'{target}_same_hour_7d_ago',
        f'{target}_same_hour_avg_7d'
    ]
    
    interaction_features = [
        'hour_month_interaction',
        'pressure_wind_interaction',
        'cloud_hour_interaction'
    ]
    
    city_features = [col for col in train_df.columns if col.startswith('city_')]
    
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   interaction_features + city_features)
    
    feature_cols = [col for col in all_features if col in train_df.columns]
    
    print(f"  Sử dụng {len(feature_cols)} features")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target]
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target]
    
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
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    def calculate_mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    mape_train = calculate_mape(y_train, y_pred_train)
    mape_val = calculate_mape(y_val, y_pred_val)
    mape_test = calculate_mape(y_test, y_pred_test)
    
    print("\n  Kết quả:")
    print(f"    Training   - R²: {r2_score(y_train, y_pred_train):.4f}, MAE: {mean_absolute_error(y_train, y_pred_train):.4f}, RMSE: {rmse_train:.4f}, MAPE: {mape_train:.2f}%")
    print(f"    Validation - R²: {r2_score(y_val, y_pred_val):.4f}, MAE: {mean_absolute_error(y_val, y_pred_val):.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%")
    print(f"    Test       - R²: {r2_score(y_test, y_pred_test):.4f}, MAE: {mean_absolute_error(y_test, y_pred_test):.4f}, RMSE: {rmse_test:.4f}, MAPE: {mape_test:.2f}%")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 features quan trọng nhất:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_r2': r2_score(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': rmse_train,
        'train_mape': mape_train,
        'val_r2': r2_score(y_val, y_pred_val),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'val_rmse': rmse_val,
        'val_mape': mape_val,
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': rmse_test,
        'test_mape': mape_test
    }

def create_advanced_features_for_others(train_df, val_df, test_df):
    print(f"\n[PHẦN 2] Đang tạo các feature nâng cao cho các model khác...")
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    for split_name, df_split in splits.items():
        df_split = df_split.sort_values(['city', 'datetime']).reset_index(drop=True)
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            
            for lag in [12, 24]:
                df_split.loc[city_mask, f'Temp_lag_{lag}'] = city_data['Temp'].shift(lag).values
            
            for window in [12, 24]:
                df_split.loc[city_mask, f'Temp_rolling_mean_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split.loc[city_mask, f'Temp_rolling_std_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
                df_split.loc[city_mask, f'Temp_rolling_max_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).max().values
                df_split.loc[city_mask, f'Temp_rolling_min_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).min().values
            
            for var in ['Pressure', 'Wind', 'Cloud', 'Rain']:
                for lag in [1, 3, 6, 12, 24]:
                    df_split.loc[city_mask, f'{var}_lag_{lag}'] = city_data[var].shift(lag).values
            
            for var in ['Pressure', 'Wind', 'Cloud', 'Rain']:
                for window in [3, 6, 12, 24]:
                    df_split.loc[city_mask, f'{var}_rolling_mean_{window}'] = city_data[var].shift(1).rolling(window=window, min_periods=1).mean().values
                    df_split.loc[city_mask, f'{var}_rolling_max_{window}'] = city_data[var].shift(1).rolling(window=window, min_periods=1).max().values
        
        df_split['has_rain'] = (df_split['Rain'] > 0).astype(int)
        df_split['has_rain_lag_1'] = df_split['has_rain'].shift(1).fillna(0).astype(int)
        df_split['has_rain_lag_3'] = df_split['has_rain'].shift(3).fillna(0).astype(int)
        df_split['has_rain_lag_6'] = df_split['has_rain'].shift(6).fillna(0).astype(int)
        
        df_split['hour_month_interaction'] = df_split['hour'] * df_split['month']
        df_split['pressure_wind_interaction'] = df_split['Pressure'] * df_split['Wind'] / 100
        df_split['cloud_hour_interaction'] = df_split['Cloud'] * df_split['hour'] / 100
        df_split['cloud_pressure_interaction'] = df_split['Cloud'] * df_split['Pressure'] / 1000
        df_split['rain_cloud_interaction'] = df_split['Rain'] * df_split['Cloud'] / 100
        df_split['temp_pressure_interaction'] = df_split['Temp'] * df_split['Pressure'] / 100
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            city_data['hour'] = city_data['datetime'].dt.hour
            
            df_split.loc[city_mask, 'Temp_same_hour_1d_ago'] = city_data.groupby('hour')['Temp'].shift(8).values
            df_split.loc[city_mask, 'Temp_same_hour_7d_ago'] = city_data.groupby('hour')['Temp'].shift(7*8).values
            df_split.loc[city_mask, 'Temp_same_hour_avg_7d'] = city_data.groupby('hour')['Temp'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            city_data['hour'] = city_data['datetime'].dt.hour
            
            df_split.loc[city_mask, 'Rain_same_hour_1d_ago'] = city_data.groupby('hour')['Rain'].shift(8).values
            df_split.loc[city_mask, 'Rain_same_hour_avg_7d'] = city_data.groupby('hour')['Rain'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            city_data['hour'] = city_data['datetime'].dt.hour
            
            df_split.loc[city_mask, 'Cloud_same_hour_1d_ago'] = city_data.groupby('hour')['Cloud'].shift(8).values
            df_split.loc[city_mask, 'Cloud_same_hour_7d_ago'] = city_data.groupby('hour')['Cloud'].shift(7*8).values
            df_split.loc[city_mask, 'Cloud_same_hour_avg_7d'] = city_data.groupby('hour')['Cloud'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        splits[split_name] = df_split
    
    print(f"  ✓ Đã tạo các feature nâng cao cho các model khác")
    return splits['train'], splits['val'], splits['test']

def train_improved_other_models(train_df, val_df, test_df):
    print("\n[PHẦN 2] Đang train các model cải tiến cho Rain, Cloud, Pressure, Wind, Gust...")
    
    target_vars = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    results = {}
    
    for target in target_vars:
        print(f"\n  Đang train model {target}...")
        
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
    
        if target == 'Pressure':
            rain_binary_features = ['has_rain_lag_3', 'has_rain_lag_6']
        elif target == 'Rain':
            rain_binary_features = ['has_rain_lag_1']
        elif target == 'Gust':
            rain_binary_features = ['has_rain_lag_3']
        elif target == 'Wind':
            rain_binary_features = ['has_rain_lag_1', 'has_rain_lag_6']
        else:
            rain_binary_features = ['has_rain_lag_1', 'has_rain_lag_3', 'has_rain_lag_6']
        
        features_to_remove = []
        if target == 'Pressure':
            features_to_remove.append('Rain_same_hour_7d_ago')
        
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
        
        city_features = [col for col in train_df.columns if col.startswith('city_')]
        
        all_features = (base_features + cyclical_features + weather_features + 
                       temp_lag_features + temp_rolling_features +
                       other_vars_lag_rolling + context_features + 
                       rain_binary_features + interaction_features + city_features)
        
        all_features = [f for f in all_features if f not in features_to_remove]
        
        feature_cols = [col for col in all_features if col in train_df.columns]
        
        print(f"    Sử dụng {len(feature_cols)} features")
        
        train_df_clean = train_df.dropna(subset=[target]).copy()
        if 'Temp_lag_24' in train_df_clean.columns:
            train_df_clean = train_df_clean[train_df_clean['Temp_lag_24'].notna()].copy()
        
        val_df_clean = val_df.dropna(subset=[target]).copy()
        if 'Temp_lag_24' in val_df_clean.columns:
            val_df_clean = val_df_clean[val_df_clean['Temp_lag_24'].notna()].copy()
        
        test_df_clean = test_df.dropna(subset=[target]).copy()
        if 'Temp_lag_24' in test_df_clean.columns:
            test_df_clean = test_df_clean[test_df_clean['Temp_lag_24'].notna()].copy()
        
        if len(train_df_clean) == 0:
            print(f"    ⚠ Không có dữ liệu để train {target}")
            continue
        
        X_train = train_df_clean[feature_cols].fillna(0)
        y_train = train_df_clean[target]
        
        X_val = val_df_clean[feature_cols].fillna(0)
        y_val = val_df_clean[target]
        
        X_test = test_df_clean[feature_cols].fillna(0)
        y_test = test_df_clean[target]
        
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
        
        print("    Đang train model...")
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        def calculate_mape(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            mask = y_true != 0
            if mask.sum() == 0:
                return np.nan
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        mape_train = calculate_mape(y_train, y_pred_train)
        mape_val = calculate_mape(y_val, y_pred_val)
        mape_test = calculate_mape(y_test, y_pred_test)
        
        print("\n    Kết quả:")
        print(f"      Training   - R²: {r2_score(y_train, y_pred_train):.4f}, MAE: {mean_absolute_error(y_train, y_pred_train):.4f}, RMSE: {rmse_train:.4f}, MAPE: {mape_train:.2f}%")
        print(f"      Validation - R²: {r2_score(y_val, y_pred_val):.4f}, MAE: {mean_absolute_error(y_val, y_pred_val):.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.2f}%")
        print(f"      Test       - R²: {r2_score(y_test, y_pred_test):.4f}, MAE: {mean_absolute_error(y_test, y_pred_test):.4f}, RMSE: {rmse_test:.4f}, MAPE: {mape_test:.2f}%")
        
        results[target] = {
            'model': model,
            'feature_cols': feature_cols,
            'train_samples': len(train_df_clean),
            'val_samples': len(val_df_clean),
            'test_samples': len(test_df_clean),
            'train_r2': r2_score(y_train, y_pred_train),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': rmse_train,
            'train_mape': mape_train,
            'val_r2': r2_score(y_val, y_pred_val),
            'val_mae': mean_absolute_error(y_val, y_pred_val),
            'val_rmse': rmse_val,
            'val_mape': mape_val,
            'test_r2': r2_score(y_test, y_pred_test),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': rmse_test,
            'test_mape': mape_test
        }
    
    return results

# ============================================================================
# THỰC THI CHÍNH
# ============================================================================

if __name__ == '__main__':
    df = load_and_preprocess_data()
    
    train_df, val_df, test_df = split_data_by_time(df)
    
    print("\n" + "="*70)
    print("PHẦN 1: MODEL NHIỆT ĐỘ")
    print("="*70)
    
    train_df_temp, val_df_temp, test_df_temp = create_advanced_features_for_temp(train_df.copy(), val_df.copy(), test_df.copy())
    
    temp_result = train_improved_temp_model(train_df_temp, val_df_temp, test_df_temp)
    
    print("\n[PHẦN 1] Đang lưu model Temp...")
    try:
        with open('weather_models_improved.pkl', 'rb') as f:
            models_data = pickle.load(f)
    except:
        models_data = {'models': {}, 'feature_cols': {}}
    
    models_data['models']['Temp_numeric'] = temp_result['model']
    models_data['feature_cols']['Temp_numeric'] = temp_result['feature_cols']
    
    with open('weather_models_improved.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    print("  ✓ Đã lưu model Temp vào weather_models_improved.pkl")
    
    print("\n" + "="*70)
    print("PHẦN 2: CÁC MODEL KHÁC (Rain, Cloud, Pressure, Wind, Gust)")
    print("="*70)
    
    train_df_others, val_df_others, test_df_others = create_advanced_features_for_others(train_df.copy(), val_df.copy(), test_df.copy())
    
    other_results = train_improved_other_models(train_df_others, val_df_others, test_df_others)
    
    print("\n[PHẦN 2] Đang lưu các model khác...")
    try:
        with open('weather_models_improved.pkl', 'rb') as f:
            models_data = pickle.load(f)
    except:
        models_data = {'models': {}, 'feature_cols': {}}
    
    for target, result in other_results.items():
        models_data['models'][f'{target}_numeric'] = result['model']
        models_data['feature_cols'][f'{target}_numeric'] = result['feature_cols']
    
    with open('weather_models_improved.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    print("  ✓ Đã lưu các model khác vào weather_models_improved.pkl")
    
    print("\n" + "="*70)
    print("TÓM TẮT - KẾT QUẢ TEST SET")
    print("="*70)
    
    print("\nTemp:")
    print(f"  Test set ({temp_result['test_samples']} samples):")
    print(f"    R²: {temp_result['test_r2']:.4f} ({temp_result['test_r2']*100:.2f}%)")
    print(f"    MAE: {temp_result['test_mae']:.4f}")
    print(f"    RMSE: {temp_result['test_rmse']:.4f}")
    print(f"    MAPE: {temp_result['test_mape']:.2f}%")
    
    for target, result in other_results.items():
        print(f"\n{target}:")
        print(f"  Test set ({result['test_samples']} samples):")
        print(f"    R²: {result['test_r2']:.4f} ({result['test_r2']*100:.2f}%)")
        print(f"    MAE: {result['test_mae']:.4f}")
        print(f"    RMSE: {result['test_rmse']:.4f}")
        print(f"    MAPE: {result['test_mape']:.2f}%")
    
    print("\n" + "="*70)
    print("HOÀN TẤT TRAINING TẤT CẢ CÁC MODEL CẢI TIẾN!")
    print("="*70)
