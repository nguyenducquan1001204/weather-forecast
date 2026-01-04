"""
Script để train models với TOÀN BỘ dữ liệu (không split) để dự báo tương lai
Sử dụng tất cả dữ liệu từ 2017 đến 2026-01-02 để train
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
print("TRAINING FINAL MODELS WITH ALL DATA (FOR FORECASTING)")
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
                # Check if database has enough data for training (at least 1000 records)
                if len(df) >= 1000:
                    print(f"  ✅ Loaded {len(df)} records from database")
                    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                    return df
                else:
                    print(f"  ⚠️  Database has only {len(df)} records (insufficient for training)")
                    print("  ⚠️  Falling back to CSV for full dataset...")
        
        print("  ⚠️  Database not found or empty. Falling back to CSV...")
    except Exception as e:
        print(f"  ⚠️  Error loading from database: {e}. Falling back to CSV...")
    
    # Fallback to CSV
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
    
    print(f"  ✅ Loaded {len(df)} records from CSV")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def create_advanced_features(df, target_col='Temp'):
    """Tạo advanced features cho TOÀN BỘ dữ liệu"""
    print(f"\n[2] Creating advanced features for {target_col}...")
    
    df = df.copy()
    df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
    
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_data = df[city_mask].copy().sort_values('datetime').reset_index(drop=True)
        
        # Lag features
        for lag in [12, 24]:
            df.loc[city_mask, f'{target_col}_lag_{lag}'] = city_data[target_col].shift(lag).values
        
        # Rolling features
        for window in [12, 24]:
            df.loc[city_mask, f'{target_col}_rolling_mean_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'{target_col}_rolling_std_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0).values
            df.loc[city_mask, f'{target_col}_rolling_max_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).max().values
            df.loc[city_mask, f'{target_col}_rolling_min_{window}'] = city_data[target_col].shift(1).rolling(window=window, min_periods=1).min().values
        
        # Lag features của Pressure, Wind, Cloud
        for lag in [1, 3, 6, 12]:
            df.loc[city_mask, f'Pressure_lag_{lag}'] = city_data['Pressure'].shift(lag).values
            df.loc[city_mask, f'Wind_lag_{lag}'] = city_data['Wind'].shift(lag).values
            df.loc[city_mask, f'Cloud_lag_{lag}'] = city_data['Cloud'].shift(lag).values
        
        # Rolling features của Pressure, Wind, Cloud
        for window in [3, 6, 12, 24]:
            df.loc[city_mask, f'Pressure_rolling_mean_{window}'] = city_data['Pressure'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Wind_rolling_mean_{window}'] = city_data['Wind'].shift(1).rolling(window=window, min_periods=1).mean().values
            df.loc[city_mask, f'Cloud_rolling_mean_{window}'] = city_data['Cloud'].shift(1).rolling(window=window, min_periods=1).mean().values
        
        # Context features
        city_data['hour'] = city_data['datetime'].dt.hour
        df.loc[city_mask, f'{target_col}_same_hour_1d_ago'] = city_data.groupby('hour')[target_col].shift(8).values
        df.loc[city_mask, f'{target_col}_same_hour_7d_ago'] = city_data.groupby('hour')[target_col].shift(7*8).values
        df.loc[city_mask, f'{target_col}_same_hour_avg_7d'] = city_data.groupby('hour')[target_col].transform(lambda x: x.shift(1).rolling(window=7*8, min_periods=1).mean()).values
    
    # Interaction features
    df['hour_month_interaction'] = df['hour'] * df['month']
    df['pressure_wind_interaction'] = df['Pressure'] * df['Wind'] / 100
    df['cloud_hour_interaction'] = df['Cloud'] * df['hour'] / 100
    
    # City encoding
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    all_cities = ['vinh', 'ha-noi', 'ho-chi-minh-city']
    for city in all_cities:
        col_name = f'city_{city}'
        if col_name not in city_dummies.columns:
            city_dummies[col_name] = 0
    
    df = pd.concat([df, city_dummies], axis=1)
    
    print(f"  ✅ Created advanced features")
    return df

def train_final_model(df, target='Temp'):
    print(f"\n[3] Training final model for {target}...")
    
    # Base features
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    
    # Cyclical features
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    
    # Weather features
    weather_features = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
    
    # Lag features của temperature
    temp_lag_features = [f'{target}_lag_{lag}' for lag in [12, 24]]
    
    # Rolling features của temperature
    temp_rolling_features = []
    for window in [12, 24]:
        temp_rolling_features.extend([
            f'{target}_rolling_mean_{window}',
            f'{target}_rolling_std_{window}',
            f'{target}_rolling_max_{window}',
            f'{target}_rolling_min_{window}'
        ])
    
    # Lag/rolling của Pressure, Wind, Cloud
    other_vars_lag_rolling = []
    for var in ['Pressure', 'Wind', 'Cloud']:
        for lag in [1, 3, 6, 12]:
            other_vars_lag_rolling.append(f'{var}_lag_{lag}')
        for window in [3, 6, 12, 24]:
            other_vars_lag_rolling.append(f'{var}_rolling_mean_{window}')
    
    # Context features
    context_features = [
        f'{target}_same_hour_1d_ago',
        f'{target}_same_hour_7d_ago',
        f'{target}_same_hour_avg_7d'
    ]
    
    # Interaction features
    interaction_features = [
        'hour_month_interaction',
        'pressure_wind_interaction',
        'cloud_hour_interaction'
    ]
    
    # City features
    city_features = [col for col in df.columns if col.startswith('city_')]
    
    # Combine all features
    all_features = (base_features + cyclical_features + weather_features + 
                   temp_lag_features + temp_rolling_features +
                   other_vars_lag_rolling + context_features + 
                   interaction_features + city_features)
    
    # Filter features that exist in dataframe
    feature_cols = [col for col in all_features if col in df.columns]
    
    print(f"  Using {len(feature_cols)} features")
    
    # Loại bỏ rows có NaN trong target hoặc features quan trọng
    df_clean = df.dropna(subset=[target]).copy()
    
    # Loại bỏ rows ở đầu (không có lag features)
    df_clean = df_clean[df_clean[f'{target}_lag_24'].notna()].copy()
    
    if len(df_clean) == 0:
        print(f"  ⚠️  Không đủ dữ liệu để train {target}")
        return None, []
    
    X = df_clean[feature_cols].fillna(0).copy()
    y = df_clean[target].copy()
    
    # Đảm bảo X là DataFrame hợp lệ (không có cột là DataFrame)
    for col in X.columns:
        if isinstance(X[col].iloc[0] if len(X) > 0 else None, pd.DataFrame):
            X[col] = X[col].apply(lambda x: x.iloc[0, 0] if isinstance(x, pd.DataFrame) else x)
    
    print(f"  Training on {len(X)} samples")
    
    # Hyperparameters (giống improved model)
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
    model.fit(X, y)
    
    # Evaluate trên toàn bộ dữ liệu
    y_pred = model.predict(X)
    
    print(f"\n  Results (on all data):")
    print(f"    MAE: {mean_absolute_error(y, y_pred):.4f}")
    print(f"    RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"    R²: {r2_score(y, y_pred):.4f}")
    
    return model, feature_cols

def train_final_hcm_model(df):
    """Train model riêng cho Hồ Chí Minh với toàn bộ dữ liệu"""
    print(f"\n[4] Training final HCM-specific model...")
    
    hcm_df = df[df['city'] == 'ho-chi-minh-city'].copy()
    print(f"  HCM data: {len(hcm_df)} records")
    
    # Tạo features (chỉ cần tạo lại nếu chưa có)
    if 'Temp_lag_24' not in hcm_df.columns:
        hcm_df = create_advanced_features(hcm_df, 'Temp')
    
    # Train model với hyperparameters tối ưu cho HCM
    print(f"\n[3] Training final HCM model for Temp...")
    
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
    
    print(f"  Using {len(feature_cols)} features")
    
    hcm_df_clean = hcm_df.dropna(subset=['Temp']).copy()
    hcm_df_clean = hcm_df_clean[hcm_df_clean['Temp_lag_24'].notna()].copy()
    
    if len(hcm_df_clean) == 0:
        print(f"  ⚠️  Không đủ dữ liệu để train HCM model")
        return None, []
    
    X = hcm_df_clean[feature_cols].fillna(0).copy()
    y = hcm_df_clean['Temp'].copy()
    
    print(f"  Training on {len(X)} samples")
    
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
    
    print("  Training model...")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    print(f"\n  Results (on all data):")
    print(f"    MAE: {mean_absolute_error(y, y_pred):.4f}")
    print(f"    RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"    R²: {r2_score(y, y_pred):.4f}")
    
    return model, feature_cols

def save_final_models(models_dict, feature_cols_dict, filename='weather_models_final.pkl'):
    print(f"\n[5] Saving final models to {filename}...")
    
    models_data = {
        'models': models_dict,
        'feature_cols': feature_cols_dict
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(models_data, f)
    
    print(f"  ✅ Saved {len(models_dict)} models to {filename}")
    print(f"  Models: {list(models_dict.keys())}")

if __name__ == '__main__':
    # Load data
    df = load_and_preprocess_data()
    
    # Tạo features cho temperature
    df = create_advanced_features(df, 'Temp')
    
    # Train models cho tất cả các target
    # Chỉ tạo advanced features cho Temp, các biến khác dùng features cơ bản
    target_vars = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
    models_dict = {}
    feature_cols_dict = {}
    
    # Train Temp model (đã có advanced features)
    model, feature_cols = train_final_model(df, 'Temp')
    if model is not None:
        models_dict['Temp_numeric'] = model
        feature_cols_dict['Temp_numeric'] = feature_cols
    
    # Train các model khác (chỉ dùng features cơ bản, không có lag/rolling của chính nó)
    for target in ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']:
        print(f"\n[3] Training final model for {target}...")
        
        # Base features
        base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
        cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                             'day_of_year_sin', 'day_of_year_cos']
        weather_features = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
        
        # Dùng lag/rolling của Temp (không dùng của chính nó)
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
        city_features = [col for col in df.columns if col.startswith('city_')]
        
        # Combine features
        all_features = (base_features + cyclical_features + weather_features + 
                       temp_lag_features + temp_rolling_features +
                       other_vars_lag_rolling + context_features + 
                       interaction_features + city_features)
        
        feature_cols = [col for col in all_features if col in df.columns]
        
        print(f"  Using {len(feature_cols)} features")
        
        # Loại bỏ rows có NaN
        df_clean = df.dropna(subset=[target]).copy()
        df_clean = df_clean[df_clean['Temp_lag_24'].notna()].copy()
        
        if len(df_clean) == 0:
            print(f"  ⚠️  Không đủ dữ liệu để train {target}")
            continue
        
        X = df_clean[feature_cols].fillna(0).copy()
        y = df_clean[target].copy()
        
        # Đảm bảo X là DataFrame hợp lệ
        for col in X.columns:
            if isinstance(X[col].iloc[0] if len(X) > 0 else None, pd.DataFrame):
                X[col] = X[col].apply(lambda x: x.iloc[0, 0] if isinstance(x, pd.DataFrame) else x)
        
        print(f"  Training on {len(X)} samples")
        
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
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        print(f"\n  Results (on all data):")
        print(f"    MAE: {mean_absolute_error(y, y_pred):.4f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        print(f"    R²: {r2_score(y, y_pred):.4f}")
        
        models_dict[f'{target}_numeric'] = model
        feature_cols_dict[f'{target}_numeric'] = feature_cols
    
    # Train model riêng cho HCM
    hcm_model, hcm_feature_cols = train_final_hcm_model(df)
    if hcm_model is not None:
        models_dict['Temp_numeric_hcm'] = hcm_model
        feature_cols_dict['Temp_numeric_hcm'] = hcm_feature_cols
    
    # Save models
    save_final_models(models_dict, feature_cols_dict, 'weather_models_final.pkl')
    
    print("\n" + "="*70)
    print("FINAL MODELS TRAINING COMPLETED!")
    print("="*70)
    print("\nModels trained on ALL available data (2017-2026-01-02)")
    print("Ready for forecasting future dates!")

