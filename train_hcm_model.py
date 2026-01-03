"""
Script để train model riêng cho Hồ Chí Minh với hyperparameters tối ưu cho khí hậu nhiệt đới
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
print("TRAINING MODEL RIENG CHO HO CHI MINH")
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
    
    return df

def create_advanced_features_after_split(train_df, val_df, test_df, target_col='Temp'):
    print(f"\n[2] Creating advanced features AFTER split...")
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
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
        df_split = pd.concat([df_split, city_dummies], axis=1)
        
        if split_name == 'train':
            train_df = df_split
        elif split_name == 'val':
            val_df = df_split
        else:
            test_df = df_split
    
    print(f"  ✅ Created advanced features")
    return train_df, val_df, test_df

def split_data_by_time(df):
    print("\n[3] Splitting data by time...")
    train_end = '2022-12-31'
    val_end = '2023-12-31'
    
    train_df = df[df['datetime'] <= train_end].copy()
    val_df = df[(df['datetime'] > train_end) & (df['datetime'] <= val_end)].copy()
    test_df = df[df['datetime'] > val_end].copy()
    
    print(f"  Train: {len(train_df)} records ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Val:   {len(val_df)} records ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"  Test:  {len(test_df)} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    return train_df, val_df, test_df

def train_hcm_temp_model(train_df, val_df, test_df):
    print("\n[4] Training HCM-specific temperature model...")
    
    target = 'Temp'
    
    base_features = ['hour', 'month', 'day_of_year', 'day_of_week', 'season', 'is_weekend']
    cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                         'day_of_year_sin', 'day_of_year_cos']
    weather_features = ['Rain', 'Cloud', 'Pressure', 'Wind', 'Gust', 'Dir']
    
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
    
    print(f"  Using {len(feature_cols)} features")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[target]
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df[target]
    
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target]
    
    print("  Hyperparameters tối ưu cho khí hậu nhiệt đới (tăng độ phức tạp để dự đoán tốt hơn nhiệt độ max)...")
    model = XGBRegressor(
        n_estimators=250,  # Tăng để học tốt hơn pattern phức tạp
        max_depth=7,       # Tăng để capture pattern nhiệt độ max
        learning_rate=0.08, # Giảm nhẹ để học kỹ hơn
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,  # Giảm để học tốt hơn pattern nhiệt đới
        gamma=0.1,           # Giảm để cho phép split phức tạp hơn
        reg_alpha=0.05,      # Giảm regularization
        reg_lambda=0.8,      # Giảm regularization
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training model...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    print("\n  Results:")
    print(f"    Training   - MAE: {mean_absolute_error(y_train, y_pred_train):.4f}, R²: {r2_score(y_train, y_pred_train):.4f}")
    print(f"    Validation - MAE: {mean_absolute_error(y_val, y_pred_val):.4f}, R²: {r2_score(y_val, y_pred_val):.4f}")
    print(f"    Test       - MAE: {mean_absolute_error(y_test, y_pred_test):.4f}, R²: {r2_score(y_test, y_pred_test):.4f}")
    
    return model, feature_cols

if __name__ == '__main__':
    df = load_and_preprocess_data()
    
    hcm_df = df[df['city'] == 'ho-chi-minh-city'].copy()
    print(f"\n  Dữ liệu Hồ Chí Minh: {len(hcm_df)} records")
    
    train_df, val_df, test_df = split_data_by_time(hcm_df)
    
    train_df, val_df, test_df = create_advanced_features_after_split(train_df, val_df, test_df, 'Temp')
    
    model, feature_cols = train_hcm_temp_model(train_df, val_df, test_df)
    
    print("\n[5] Saving HCM model...")
    try:
        with open('weather_models_improved.pkl', 'rb') as f:
            models_data = pickle.load(f)
    except:
        models_data = {'models': {}, 'feature_cols': {}}
    
    models_data['models']['Temp_numeric_hcm'] = model
    models_data['feature_cols']['Temp_numeric_hcm'] = feature_cols
    
    with open('weather_models_improved.pkl', 'wb') as f:
        pickle.dump(models_data, f)
    
    print("  ✅ Saved HCM model to weather_models_improved.pkl")
    print("\n" + "="*70)
    print("HCM MODEL TRAINING COMPLETED!")
    print("="*70)

