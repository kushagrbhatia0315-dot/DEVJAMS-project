import os
import joblib
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier
def train_and_predict(X_train, y_train, X_test, save_path: str):
    """Trains the algorithm and saves it to disk."""
    print("--> [4/5] Training Random Forest with Merged Data...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model.predict(X_test)
def predict_fall_2026_scenario(save_path: str, training_columns: list, merged_df: pd.DataFrame):
    """Generates Fall 2026 forecasts using real historical state/month metrics."""
    print("--> Forecasting Fall 2026 Scenarios...")
    model = joblib.load(save_path)
    states = ['CA', 'TX', 'FL', 'OR', 'WA', 'CO', 'AZ']
    months = [9, 10, 11]
    scenarios = []
    state_month_stats = merged_df.groupby(['STATE', 'MONTH']).agg({
        'FIRE_SIZE': 'median',
        'MONTHLY_STORMS': 'mean'
    }).reset_index()
    overall_fire_size = merged_df['FIRE_SIZE'].median()
    overall_storms = merged_df['MONTHLY_STORMS'].mean()
    for state in states:
        for month in months:
            match = state_month_stats[(state_month_stats['STATE'] == state) & (state_month_stats['MONTH'] == month)]
            if not match.empty:
                avg_fire_size = round(match['FIRE_SIZE'].values[0], 2)
                avg_storms = round(match['MONTHLY_STORMS'].values[0], 1)
            else:
                avg_fire_size = round(overall_fire_size, 2)
                avg_storms = round(overall_storms, 1)
            scenarios.append({
                'FIRE_YEAR': 2026,
                'FIRE_SIZE': avg_fire_size,
                'MONTH': month,
                'MONTHLY_STORMS': avg_storms,
                'STATE': state
            })       
    scenario_df = pd.DataFrame(scenarios)
    X_features = scenario_df.drop(columns=['FIRE_SIZE', 'MONTHLY_STORMS'], errors='ignore')
    X_scenario = pd.get_dummies(X_features)
    X_scenario['FIRE_SIZE'] = scenario_df['FIRE_SIZE']
    X_scenario['MONTHLY_STORMS'] = scenario_df['MONTHLY_STORMS']
    X_scenario = X_scenario.reindex(columns=training_columns, fill_value=0)
    scenario_df['PREDICTED_CAUSE'] = model.predict(X_scenario)
    return scenario_df
