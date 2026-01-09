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
print("TRAINING ALL MODELS RIENG CHO HO CHI MINH (CÁCH MỚI)")
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

def create_advanced_features_for_hcm(train_df, val_df, test_df):
    print(f"\n[3] Đang tạo các feature nâng cao cho các model HCM...")
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    for split_name, df_split in splits.items():
        df_split = df_split.sort_values('datetime').reset_index(drop=True)
        
        for lag in [12, 24]:
            df_split[f'Temp_lag_{lag}'] = df_split['Temp'].shift(lag).values
        
        for window in [12, 24]:
            df_split[f'Temp_rolling_mean_{window}'] = df_split['Temp'].shift(1).rolling(window=window, min_periods=1).mean().values
            df_split[f'Temp_rolling_std_{window}'] = df_split['Temp'].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
            df_split[f'Temp_rolling_max_{window}'] = df_split['Temp'].shift(1).rolling(window=window, min_periods=1).max().values
            df_split[f'Temp_rolling_min_{window}'] = df_split['Temp'].shift(1).rolling(window=window, min_periods=1).min().values
        
        for var in ['Pressure', 'Wind', 'Cloud', 'Rain']:
            for lag in [1, 3, 6, 12, 24]:
                df_split[f'{var}_lag_{lag}'] = df_split[var].shift(lag).values
        
        for var in ['Pressure', 'Wind', 'Cloud', 'Rain']:
            for window in [3, 6, 12, 24]:
                df_split[f'{var}_rolling_mean_{window}'] = df_split[var].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split[f'{var}_rolling_max_{window}'] = df_split[var].shift(1).rolling(window=window, min_periods=1).max().values
        
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
        
        df_split['hour'] = df_split['datetime'].dt.hour
        df_split['Temp_same_hour_1d_ago'] = df_split.groupby('hour')['Temp'].shift(8).values
        df_split['Temp_same_hour_7d_ago'] = df_split.groupby('hour')['Temp'].shift(7*8).values
        df_split['Temp_same_hour_avg_7d'] = df_split.groupby('hour')['Temp'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        df_split['Rain_same_hour_1d_ago'] = df_split.groupby('hour')['Rain'].shift(8).values
        df_split['Rain_same_hour_avg_7d'] = df_split.groupby('hour')['Rain'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        df_split['Cloud_same_hour_1d_ago'] = df_split.groupby('hour')['Cloud'].shift(8).values
        df_split['Cloud_same_hour_7d_ago'] = df_split.groupby('hour')['Cloud'].shift(7*8).values
        df_split['Cloud_same_hour_avg_7d'] = df_split.groupby('hour')['Cloud'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
        
        splits[split_name] = df_split
    
    print(f"  ✓ Đã tạo các feature nâng cao cho các model HCM")
    return splits['train'], splits['val'], splits['test']

def train_hcm_model(train_df, val_df, test_df, target):
    print(f"\n[4] Đang train model HCM cho {target}...")
    
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
                context_features.extend(['Rain_same_hour_1d_ago', 'Rain_same_hour_avg_7d'])
        
        if target != 'Cloud':
            context_features.extend(['Cloud_same_hour_1d_ago', 'Cloud_same_hour_7d_ago', 'Cloud_same_hour_avg_7d'])
        
        if target == 'Rain':
            context_features.extend(['Rain_same_hour_1d_ago', 'Rain_same_hour_avg_7d'])
        
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
    
    features_to_remove = []
    features_to_remove.append('Rain_same_hour_7d_ago')
    
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   rain_binary_features + interaction_features)
    
    all_features = [f for f in all_features if f not in features_to_remove]
    
    feature_cols = [col for col in all_features if col in train_df.columns]
    
    print(f"  Sử dụng {len(feature_cols)} features")
    
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
        print(f"  ⚠ Không có dữ liệu để train {target}")
        return None
    
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
    elif target == 'Temp':
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
    
    return {
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

if __name__ == '__main__':
    df = load_and_preprocess_data()
    
    hcm_df = df[df['city'] == 'ho-chi-minh-city'].copy()
    print(f"\n  Dữ liệu Hồ Chí Minh: {len(hcm_df)} bản ghi")
    
    train_df, val_df, test_df = split_data_by_time(hcm_df)
    
    train_df, val_df, test_df = create_advanced_features_for_hcm(train_df, val_df, test_df)
    
    target_vars = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    results = {}
    
    for target in target_vars:
        result = train_hcm_model(train_df, val_df, test_df, target)
        if result is not None:
            results[target] = result
    
    print("\n[5] Đang lưu các model HCM...")
    try:
        with open('weather_models_improved.pkl', 'rb') as f:
            models_data = pickle.load(f)
    except:
        models_data = {'models': {}, 'feature_cols': {}}
    
    for target, result in results.items():
        models_data['models'][f'{target}_numeric_hcm'] = result['model']
        models_data['feature_cols'][f'{target}_numeric_hcm'] = result['feature_cols']
    
    with open('weather_models_improved.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    print("  ✓ Đã lưu các model HCM vào weather_models_improved.pkl")
    
    print("\n" + "="*70)
    print("TÓM TẮT - KẾT QUẢ TEST SET (HCM)")
    print("="*70)
    for target, result in results.items():
        print(f"\n{target}:")
        print(f"  Test set ({result['test_samples']} samples):")
        print(f"    R²: {result['test_r2']:.4f} ({result['test_r2']*100:.2f}%)")
        print(f"    MAE: {result['test_mae']:.4f}")
        print(f"    RMSE: {result['test_rmse']:.4f}")
        print(f"    MAPE: {result['test_mape']:.2f}%")
    
    print("\n" + "="*70)
    print("HOÀN TẤT TRAINING TẤT CẢ CÁC MODEL HCM!")
    print("="*70)
