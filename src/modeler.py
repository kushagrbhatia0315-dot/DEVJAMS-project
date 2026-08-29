import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(X_train, y_train, X_test, save_path: str):
    """Trains the algorithm and saves it to disk."""
    print("--> [4/5] Training Random Forest with Merged Data...")
    
    # Removed random_state to ensure varying tree building
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    
    return model.predict(X_test)
