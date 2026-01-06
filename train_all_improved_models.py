"""
Script để train tất cả các mô hình (Rain, Cloud, Pressure, Wind, Gust) với train/val/test split
"""
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
print("TRAINING ALL IMPROVED MODELS WITH TRAIN/VAL/TEST SPLIT")
print("="*70)

def load_and_preprocess_data(file_path=None):
    """Load data from SQLite database (fallback to CSV if database doesn't exist)"""
    print("\n[1] Loading and preprocessing data...")
    
    try:
        from database import load_data_from_db
        import os
        
        if os.path.exists('weather.db'):
            df = load_data_from_db()
            if len(df) > 0:
                print(f"  ✅ Loaded {len(df)} records from database")
                return df
        
        print("  ⚠️  Database not found or empty. Falling back to CSV...")
    except Exception as e:
        print(f"  ⚠️  Error loading from database: {e}. Falling back to CSV...")
    
    # Fallback to CSV
    if file_path is None:
        file_path = 'weather_all_cities.csv'
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'])
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    # Làm sạch cột gốc và giữ nguyên tên, không tạo cột mới
    df['Temp'] = df['Temp'].str.replace(' °c', '').str.replace('°c', '').astype(float)
    df['Rain'] = df['Rain'].str.replace('mm', '').astype(float)
    df['Cloud'] = df['Cloud'].str.replace('%', '').astype(float)
    df['Pressure'] = df['Pressure'].str.replace(' mb', '').astype(float)
    df['Wind'] = df['Wind'].str.replace(' km/h', '').astype(float)
    df['Gust'] = df['Gust'].str.replace(' km/h', '').astype(float)
    
    # Tạo các features cơ bản
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['season'] = df['month'].apply(lambda x: (x % 12) // 3 + 1)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # City encoding
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    print(f"  ✅ Loaded {len(df)} records from CSV")
    
    return df

def create_advanced_features_after_split(train_df, val_df, test_df):
    """Tạo features SAU KHI split, đảm bảo lag chỉ dùng quá khứ"""
    print(f"\n[2] Creating advanced features AFTER split (with lag/rolling of Temp, no leakage)...")
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    # Tạo features cho từng split riêng biệt
    for df_split, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df_split = df_split.sort_values(['city', 'datetime']).reset_index(drop=True)
        
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            
            # 1. Lag features của Temp (chỉ dùng lag xa: 12, 24)
            for lag in [12, 24]:
                df_split.loc[city_mask, f'Temp_lag_{lag}'] = city_data['Temp'].shift(lag).values
            
            # 2. Rolling features của Temp (chỉ dùng window lớn: 12, 24)
            for window in [12, 24]:
                df_split.loc[city_mask, f'Temp_rolling_mean_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).mean().values
                df_split.loc[city_mask, f'Temp_rolling_std_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
                df_split.loc[city_mask, f'Temp_rolling_max_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).max().values
                df_split.loc[city_mask, f'Temp_rolling_min_{window}'] = city_data['Temp'].shift(1).rolling(window=window, min_periods=1).min().values
            
            # 3. Lag features của Pressure, Wind, Cloud
            for var in ['Pressure', 'Wind', 'Cloud']:
                for lag in [1, 3, 6, 12]:
                    df_split.loc[city_mask, f'{var}_lag_{lag}'] = city_data[var].shift(lag).values
            
            # 4. Rolling features của Pressure, Wind, Cloud
            for var in ['Pressure', 'Wind', 'Cloud']:
                for window in [3, 6, 12, 24]:
                    df_split.loc[city_mask, f'{var}_rolling_mean_{window}'] = city_data[var].shift(1).rolling(window=window, min_periods=1).mean().values
        
        # Interaction features
        df_split['hour_month_interaction'] = df_split['hour'] * df_split['month']
        df_split['pressure_wind_interaction'] = df_split['Pressure'] * df_split['Wind'] / 100
        df_split['cloud_hour_interaction'] = df_split['Cloud'] * df_split['hour'] / 100
        
        # Context features của Temp
        city_data = df_split.sort_values(['city', 'datetime']).reset_index(drop=True)
        for city in df_split['city'].unique():
            city_mask = df_split['city'] == city
            city_data = df_split[city_mask].copy().sort_values('datetime').reset_index(drop=True)
            city_data['hour'] = city_data['datetime'].dt.hour
            
            df_split.loc[city_mask, 'Temp_same_hour_1d_ago'] = city_data.groupby('hour')['Temp'].shift(8).values
            df_split.loc[city_mask, 'Temp_same_hour_7d_ago'] = city_data.groupby('hour')['Temp'].shift(7*8).values
            df_split.loc[city_mask, 'Temp_same_hour_avg_7d'] = city_data.groupby('hour')['Temp'].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
    
    print(f"  ✅ Created advanced features")
    return train_df, val_df, test_df

def split_data_by_time(df):
    print("\n[3] Splitting data by time...")
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    val_start = '2021-01-01'
    
    train_df = df[df['datetime'] <= train_end].copy()
    test_df = df[df['datetime'] >= test_start].copy()
    val_df = train_df[train_df['datetime'] >= val_start].copy()
    train_df_only = train_df[train_df['datetime'] < val_start].copy()
    
    print(f"  - Training: {len(train_df_only)} records")
    print(f"  - Validation: {len(val_df)} records")
    print(f"  - Test: {len(test_df)} records")
    
    return train_df_only, val_df, test_df

def train_improved_model(train_df, val_df, test_df, target):
    print(f"\n[4] Training improved model for {target}...")
    
    # Base features
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    
    # Cyclical features
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    
    # Weather features (không bao gồm chính target)
    weather_features = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
    weather_features = [f for f in weather_features if f != target]
    
    # Lag/rolling của Temp (không dùng của chính target)
    temp_lag_features = ['Temp_lag_12', 'Temp_lag_24']
    temp_rolling_features = ['Temp_rolling_mean_12', 'Temp_rolling_std_12', 
                             'Temp_rolling_max_12', 'Temp_rolling_min_12',
                             'Temp_rolling_mean_24', 'Temp_rolling_std_24',
                             'Temp_rolling_max_24', 'Temp_rolling_min_24']
    
    # Lag/rolling của Pressure, Wind, Cloud (không dùng của chính target)
    other_vars_lag_rolling = []
    for var in ['Pressure', 'Wind', 'Cloud']:
        if var != target:
            for lag in [1, 3, 6, 12]:
                other_vars_lag_rolling.append(f'{var}_lag_{lag}')
            for window in [3, 6, 12, 24]:
                other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
    
    # Context features của Temp
    context_features = ['Temp_same_hour_1d_ago', 'Temp_same_hour_7d_ago', 'Temp_same_hour_avg_7d']
    
    # Interaction features
    interaction_features = ['hour_month_interaction', 'pressure_wind_interaction', 'cloud_hour_interaction']
    
    # City features
    city_features = [col for col in train_df.columns if col.startswith('city_')]
    
    # Combine features
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   interaction_features + city_features)
    
    feature_cols = [col for col in all_features if col in train_df.columns]
    
    print(f"  Using {len(feature_cols)} features")
    
    # Loại bỏ rows có NaN
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
        print(f"  ⚠️  Không đủ dữ liệu để train {target}")
        return None, []
    
    X_train = train_df_clean[feature_cols].fillna(0)
    y_train = train_df_clean[target]
    
    X_val = val_df_clean[feature_cols].fillna(0)
    y_val = val_df_clean[target]
    
    X_test = test_df_clean[feature_cols].fillna(0)
    y_test = test_df_clean[target]
    
    # Hyperparameters
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
    
    print("  Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate MAPE
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
    
    print("\n  Results:")
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
    # Load data
    df = load_and_preprocess_data()
    
    # Split data TRƯỚC
    train_df, val_df, test_df = split_data_by_time(df)
    
    # Create advanced features SAU KHI split (tránh data leakage)
    train_df, val_df, test_df = create_advanced_features_after_split(train_df, val_df, test_df)
    
    # Train models cho tất cả các target (trừ Temp vì đã có script riêng)
    target_vars = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    results = {}
    
    for target in target_vars:
        result = train_improved_model(train_df, val_df, test_df, target)
        if result is not None:
            results[target] = result
    
    # Save models
    print("\n[5] Saving improved models...")
    try:
        with open('weather_models_improved.pkl', 'rb') as f:
            models_data = pickle.load(f)
    except:
        models_data = {'models': {}, 'feature_cols': {}}
    
    for target, result in results.items():
        models_data['models'][f'{target}_numeric'] = result['model']
        models_data['feature_cols'][f'{target}_numeric'] = result['feature_cols']
    
    with open('weather_models_improved.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    print("  ✅ Saved improved models to weather_models_improved.pkl")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - TEST SET RESULTS (Đánh giá khách quan)")
    print("="*70)
    for target, result in results.items():
        print(f"\n{target}:")
        print(f"  Test set ({result['test_samples']} mẫu):")
        print(f"    R²: {result['test_r2']:.4f} ({result['test_r2']*100:.2f}%)")
        print(f"    MAE: {result['test_mae']:.4f}")
        print(f"    RMSE: {result['test_rmse']:.4f}")
        print(f"    MAPE: {result['test_mape']:.2f}%")
    
    print("\n" + "="*70)
    print("ALL IMPROVED MODELS TRAINING COMPLETED!")
    print("="*70)

