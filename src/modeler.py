import os
import joblib
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(X_train, y_train, X_test, save_path: str):
    """Trains the algorithm and saves it to disk."""
    print("--> [4/5] Training Random Forest with Merged Data...")
    model = RandomForestClassifier(n_estimators=100,max_depth=15,n_jobs=-1)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model.predict(X_test)

def predict_fall_2026_scenario(save_path: str, training_columns: list):
    print("--> Forecasting Fall 2026 Scenarios...")
    model = joblib.load(save_path)
    states = ['CA', 'TX', 'FL', 'OR', 'WA', 'CO', 'AZ']
    months = [9, 10, 11]
    scenarios = []
    for state in states:
        for month in months:
            scenarios.append({'FIRE_YEAR': 2026,'FIRE_SIZE': 10.0,    'MONTH': month,'MONTHLY_STORMS': 5,  'STATE': state})
            
    scenario_df = pd.DataFrame(scenarios)
    X_scenario = pd.get_dummies(scenario_df)
    X_scenario = X_scenario.reindex(columns=training_columns, fill_value=0)
    scenario_df['PREDICTED_CAUSE'] = model.predict(X_scenario)
    
    return scenario_df
