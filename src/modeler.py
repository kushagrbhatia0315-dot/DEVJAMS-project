import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def train_and_predict(X_train, y_train, X_test, save_path: str):
    print("--> [4/5] Training XGBoost Binary Classifier...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Standard balancing for Binary models
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model = XGBClassifier(
        n_estimators=300, 
        max_depth=7, 
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
        n_jobs=-1, 
        random_state=42,
        eval_metric='logloss', # Optimized for Binary target
        tree_method='hist'
    )
    
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'model': model, 'encoder': le}, save_path)
    
    predictions_encoded = model.predict(X_test)
    predictions_strings = le.inverse_transform(predictions_encoded)
    
    return predictions_strings

def predict_fall_2026_scenario(save_path: str, training_columns: list, merged_df: pd.DataFrame):
    print("--> Forecasting Fall 2026 Scenarios...")
    artifacts = joblib.load(save_path)
    model = artifacts['model']
    le = artifacts['encoder']
    
    states = ['CA', 'TX', 'FL', 'OR', 'WA', 'CO', 'AZ']
    months = [9, 10, 11]
    approx_doy = {9: 258, 10: 288, 11: 319}
    scenarios = []
    
    state_stats = merged_df.groupby('STATE').agg({
        'LOG_FIRE_SIZE': 'median',
        'MONTHLY_STORMS': 'mean',
        'LAT_ROUNDED': 'median',
        'LON_ROUNDED': 'median',
        'SPATIAL_INTERACT': 'median'
    }).reset_index()
    
    overall_fire_size = merged_df['LOG_FIRE_SIZE'].median()
    overall_storms = merged_df['MONTHLY_STORMS'].mean()
    overall_lat = merged_df['LAT_ROUNDED'].median()
    overall_lon = merged_df['LON_ROUNDED'].median()
    overall_spatial = merged_df['SPATIAL_INTERACT'].median()

    for state in states:
        for month in months:
            doy = approx_doy[month]
            match = state_stats[state_stats['STATE'] == state]
            
            fire_sz = match['LOG_FIRE_SIZE'].values[0] if not match.empty else overall_fire_size
            strms = match['MONTHLY_STORMS'].values[0] if not match.empty else overall_storms
            lat = match['LAT_ROUNDED'].values[0] if not match.empty else overall_lat
            lon = match['LON_ROUNDED'].values[0] if not match.empty else overall_lon
            spatial = match['SPATIAL_INTERACT'].values[0] if not match.empty else overall_spatial
            
            scenarios.append({
                'STATE': state,
                'DOY_SIN': np.sin(2 * np.pi * doy / 365.25),
                'DOY_COS': np.cos(2 * np.pi * doy / 365.25),
                'MONTH_SIN': np.sin(2 * np.pi * month / 12.0),
                'MONTH_COS': np.cos(2 * np.pi * month / 12.0),
                'LAT_ROUNDED': lat,
                'LON_ROUNDED': lon,
                'SPATIAL_INTERACT': spatial,
                'LOG_FIRE_SIZE': fire_sz,
                'MONTHLY_STORMS': strms,
                'STORM_FIRE_INTERACTION': fire_sz * np.log1p(strms)
            })
                
    scenario_df = pd.DataFrame(scenarios)
    X_scenario = pd.get_dummies(scenario_df)
    X_scenario = X_scenario.reindex(columns=training_columns, fill_value=0)
    
    preds_encoded = model.predict(X_scenario)
    scenario_df['PREDICTED_CAUSE'] = le.inverse_transform(preds_encoded)
    
    return scenario_df
