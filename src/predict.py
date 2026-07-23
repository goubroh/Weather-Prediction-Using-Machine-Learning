import os
import joblib
import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data

def predict_actual_vs_predicted(custom_input_dict=None, test_sample_index=0):
    """
    Generates actual vs predicted comparison across all trained models.
    Supports either a custom input dictionary or pulling directly from the test dataset.
    """
    model_path = "models/trained_models.joblib"
    scaler_path = "models/scaler.joblib"

    # 1. Check Artifact Existence
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model or Scaler files missing in 'models/'. "
            "Please run 'python src/train.py' first."
        )

    # 2. Load Models & Scaler
    models = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Construct Input DataFrame & Determine Actual Ground Truth
    if custom_input_dict is not None:
        # Custom dict mode
        date_str = custom_input_dict.get('date', '2024-11-20')
        date_obj = pd.to_datetime(date_str)
        
        precip = custom_input_dict.get('precipitation', 15.2)
        temp_max = custom_input_dict.get('temp_max', 12.0)
        temp_min = custom_input_dict.get('temp_min', 6.5)
        wind = custom_input_dict.get('wind', 5.1)

        # Ground truth mapping ('rain' / 1 -> RAIN, 'sun' / 0 -> NO RAIN)
        raw_actual = custom_input_dict.get('actual_weather', 'rain')
        if isinstance(raw_actual, str):
            actual_target = 1 if raw_actual.lower() == 'rain' else 0
        else:
            actual_target = int(raw_actual)

        month = date_obj.month
        day_of_year = date_obj.dayofyear

        input_dict = {
            'precipitation': precip, 'temp_max': temp_max, 'temp_min': temp_min, 'wind': wind,
            'sin_month': np.sin(2 * np.pi * month / 12),
            'cos_month': np.cos(2 * np.pi * month / 12),
            'sin_day': np.sin(2 * np.pi * day_of_year / 365.25),
            'cos_day': np.cos(2 * np.pi * day_of_year / 365.25),
            'temp_range': temp_max - temp_min,
            'temp_avg': (temp_max + temp_min) / 2.0,
            'precip_lag_1': precip, 'temp_max_lag_1': temp_max,
            'temp_min_lag_1': temp_min, 'wind_lag_1': wind,
            'precip_lag_2': precip, 'temp_max_lag_2': temp_max,
            'temp_min_lag_2': temp_min, 'wind_lag_2': wind,
            'precip_lag_3': precip, 'temp_max_lag_3': temp_max,
            'temp_min_lag_3': temp_min, 'wind_lag_3': wind,
            'precip_roll_3': precip, 'temp_max_roll_3': temp_max,
            'wind_roll_3': wind
        }
        
        input_df = pd.DataFrame([input_dict])
        if hasattr(scaler, 'feature_names_in_'):
            input_df = input_df[scaler.feature_names_in_]
            
        scaled_array = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_array, columns=input_df.columns)

    else:
        # Load from actual test dataset split
        _, X_test, _, y_test = load_and_preprocess_data(save_scaler=False)
        sample_idx = min(test_sample_index, len(X_test) - 1)
        
        scaled_input_df = X_test.iloc[[sample_idx]]
        actual_target = y_test.iloc[sample_idx]

    actual_str = 'RAIN 🌧️' if actual_target == 1 else 'NO RAIN ☀️'

    # 4. Generate Predictions & Format Comparison Table
    comparison_results = []
    for name, model in models.items():
        pred_class = model.predict(scaled_input_df)[0]
        pred_str = 'RAIN 🌧️' if pred_class == 1 else 'NO RAIN ☀️'
        
        # Match Check
        match_status = "✅ Correct" if pred_class == actual_target else "❌ Incorrect"

        # Probability Score
        if hasattr(model, 'predict_proba'):
            rain_prob = model.predict_proba(scaled_input_df)[0][1]
            prob_str = f"{rain_prob * 100:.2f}%"
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(scaled_input_df)[0]
            prob_str = f"Score: {score:.2f}"
        else:
            prob_str = "N/A"

        comparison_results.append({
            'Algorithm': name,
            'Actual Value': actual_str,
            'Predicted Value': pred_str,
            'Match Status': match_status,
            'Rain Probability': prob_str
        })

    return pd.DataFrame(comparison_results), actual_str


if __name__ == "__main__":
    print("\n" + "="*65)
    print("      ACTUAL VS PREDICTED WEATHER INFERENCE COMPARISON      ")
    print("="*65)

    # Example Sample Input with Known Ground Truth
    sample_weather_input = {
        'date': '2024-11-20',
        'precipitation': 15.2,
        'temp_max': 12.0,
        'temp_min': 6.5,
        'wind': 5.1,
        'actual_weather': 'rain'  # Ground truth label
    }

    print("\nInput Data Parameters:")
    for k, v in sample_weather_input.items():
        print(f"  • {k}: {v}")

    results_df, actual_ground_truth = predict_actual_vs_predicted(sample_weather_input)

    print(f"\nGround Truth Target: {actual_ground_truth}")
    print("\nModel Prediction Results:")
    print("-" * 65)
    print(results_df.to_string(index=False))
    print("-" * 65)