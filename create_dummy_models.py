import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def create_model_placeholders():
    """Creates dummy model and scaler for development and testing."""
    os.makedirs('models', exist_ok=True)
    
    # 1. Create a dummy StandardScaler
    # Our feature vector has 73 features
    scaler = StandardScaler()
    dummy_data = np.random.rand(10, 73)
    scaler.fit(dummy_data)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Created dummy scaler.pkl")
    
    # 2. Create a dummy RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    X = np.random.rand(20, 73)
    y = np.random.randint(0, 2, 20) # 0: HUMAN, 1: AI_GENERATED
    model.fit(X, y)
    joblib.dump(model, 'models/model.pkl')
    print("Created dummy model.pkl")

if __name__ == "__main__":
    create_model_placeholders()
