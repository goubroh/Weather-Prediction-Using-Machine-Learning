import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess_data(file_path="data/raw/seattle-weather.csv"):
    df = pd.read_csv(file_path)

    # Target
    df['target'] = df['weather'].apply(lambda x: 1 if x == 'rain' else 0)

    # Date Features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        month = df['date'].dt.month
        day_of_year = df['date'].dt.dayofyear
        
        df['sin_month'] = np.sin(2 * np.pi * month / 12)
        df['cos_month'] = np.cos(2 * np.pi * month / 12)
        df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Domain Features
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2.0

    # Lags
    for lag in [1, 2, 3]:
        df[f'precip_lag_{lag}'] = df['precipitation'].shift(lag)
        df[f'temp_max_lag_{lag}'] = df['temp_max'].shift(lag)
        df[f'temp_min_lag_{lag}'] = df['temp_min'].shift(lag)
        df[f'wind_lag_{lag}'] = df['wind'].shift(lag)

    # Rolling Means
    df['precip_roll_3'] = df['precipitation'].shift(1).rolling(window=3).mean()
    df['temp_max_roll_3'] = df['temp_max'].shift(1).rolling(window=3).mean()
    df['wind_roll_3'] = df['wind'].shift(1).rolling(window=3).mean()

    # Drop non-features
    drop_cols = ['weather', 'target']
    if 'date' in df.columns:
        drop_cols.append('date')

    X = df.drop(columns=drop_cols)
    y = df['target']

    X = X.bfill().ffill()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save fitted scaler
    joblib.dump(scaler, "models/scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()