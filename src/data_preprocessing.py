import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path="data/raw/seattle-weather.csv", save_scaler=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset missing at '{file_path}'. Please verify path.")

    df = pd.read_csv(file_path)

    # 1. Target Encoding
    df['target'] = df['weather'].apply(lambda x: 1 if x == 'rain' else 0)

    # 2. Date & Cyclical Features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        month = df['date'].dt.month
        day_of_year = df['date'].dt.dayofyear
        
        df['sin_month'] = np.sin(2 * np.pi * month / 12)
        df['cos_month'] = np.cos(2 * np.pi * month / 12)
        df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 3. Domain Specific Features
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2.0

    # 4. Lag Features
    for lag in [1, 2, 3]:
        df[f'precip_lag_{lag}'] = df['precipitation'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min'].shift(lag)
        df[f'wind_lag_{lag}'] = df['wind'].shift(lag)

    # 5. Rolling Window Features
    df['precip_roll_3'] = df['precipitation'].shift(1).rolling(window=3).mean()
    df['temp_max_roll_3'] = df['temp_max'].shift(1).rolling(window=3).mean()
    df['wind_roll_3'] = df['wind'].shift(1).rolling(window=3).mean()

    # 6. Drop NaNs created by Lags/Rolling (Prevents future-data leakage)
    df = df.dropna().reset_index(drop=True)

    # 7. Drop Non-Feature Columns
    drop_cols = ['weather', 'target']
    if 'date' in df.columns:
        drop_cols.append('date')

    X = df.drop(columns=drop_cols)
    y = df['target']

    # 8. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 9. Scaler (Fit ONLY on Train)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    if save_scaler:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()