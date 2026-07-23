import joblib
import pandas as pd
import numpy as np

def predict_single_input(custom_input_dict):
    """
    Takes a dictionary of single-day raw weather inputs, computes
    lag/cyclical features, and returns predictions across all models.
    """
    # 1. Load trained models and fitted scaler
    models = joblib.load("models/trained_models.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # 2. Extract input values or fallback to default Seattle weather medians
    date_str = custom_input_dict.get('date', '2024-10-15')
    date_obj = pd.to_datetime(date_str)
    
    precip = custom_input_dict.get('precipitation', 12.5)  # in mm
    temp_max = custom_input_dict.get('temp_max', 15.0)     # in °C
    temp_min = custom_input_dict.get('temp_min', 8.0)      # in °C
    wind = custom_input_dict.get('wind', 4.5)              # in m/s

    # 3. Derive Date & Domain Features
    month = date_obj.month
    day_of_year = date_obj.dayofyear

    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)

    temp_range = temp_max - temp_min
    temp_avg = (temp_max + temp_min) / 2.0

    # 4. Derive Lags / Rolling Averages (Using present values as proxy for single inference)
    precip_lag_1, precip_lag_2, precip_lag_3 = precip, precip, precip
    temp_max_lag_1, temp_max_lag_2, temp_max_lag_3 = temp_max, temp_max, temp_max
    temp_min_lag_1, temp_min_lag_2, temp_min_lag_3 = temp_min, temp_min, temp_min
    wind_lag_1, wind_lag_2, wind_lag_3 = wind, wind, wind

    precip_roll_3 = precip
    temp_max_roll_3 = temp_max
    wind_roll_3 = wind

    # 5. Build Input Dataframe matching training column alignment
    input_df = pd.DataFrame([{
        'precipitation': precip, 'temp_max': temp_max, 'temp_min': temp_min, 'wind': wind,
        'sin_month': sin_month, 'cos_month': cos_month,
        'sin_day': sin_day, 'cos_day': cos_day,
        'temp_range': temp_range, 'temp_avg': temp_avg,
        'precip_lag_1': precip_lag_1, 'temp_max_lag_1': temp_max_lag_1,
        'temp_min_lag_1': temp_min_lag_1, 'wind_lag_1': wind_lag_1,
        'precip_lag_2': precip_lag_2, 'temp_max_lag_2': temp_max_lag_2,
        'temp_min_lag_2': temp_min_lag_2, 'wind_lag_2': wind_lag_2,
        'precip_lag_3': precip_lag_3, 'temp_max_lag_3': temp_max_lag_3,
        'temp_min_lag_3': temp_min_lag_3, 'wind_lag_3': wind_lag_3,
        'precip_roll_3': precip_roll_3, 'temp_max_roll_3': temp_max_roll_3,
        'wind_roll_3': wind_roll_3
    }])

    # 6. Scale Input Features
    scaled_input = scaler.transform(input_df)

    # 7. Collect Individual Predictions & Probabilities
    comparison_results = []

    for name, model in models.items():
        pred_class = model.predict(scaled_input)[0]
        
        if hasattr(model, 'predict_proba'):
            rain_prob = model.predict_proba(scaled_input)[0][1]
        else:
            rain_prob = model.decision_function(scaled_input)[0]

        comparison_results.append({
            'Algorithm': name,
            'Prediction': 'RAIN 🌧️' if pred_class == 1 else 'NO RAIN ☀️',
            'Rain Probability': f"{rain_prob * 100:.2f}%" if hasattr(model, 'predict_proba') else 'N/A'
        })

    return pd.DataFrame(comparison_results)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("      CUSTOM WEATHER INFERENCE & MODEL COMPARISON      ")
    print("="*55)

    # --- DEFINE YOUR CUSTOM INPUT DATA HERE ---
    sample_weather_input = {
        'date': '2024-11-20',
        'precipitation': 15.2,  # mm of precipitation
        'temp_max': 12.0,       # High temp in °C
        'temp_min': 6.5,        # Low temp in °C
        'wind': 5.1             # Wind speed
    }

    print("\nInput Parameters:")
    for k, v in sample_weather_input.items():
        print(f"  • {k}: {v}")

    # Get comparison
    results_df = predict_single_input(sample_weather_input)

    print("\nIndividual Model Predictions Comparison:")
    print("-" * 55)
    print(results_df.to_string(index=False))
    print("-" * 55)