import joblib
import os

class ModelHandler:
    def __init__(self, model_path='models/model.pkl', scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        """Loads the saved models from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def predict(self, features):
        """Runs inference on extracted features."""
        if self.model is None or self.scaler is None:
            return None, "Model or Scaler not loaded. Please ensure .pkl files are in models/ directory."

        try:
            # Print features for debugging
            print(f"DEBUG: Features Shape: {features.shape}")
            print(f"DEBUG: First 5 features: {features[0][:5]}")
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            # Assuming 0 is HUMAN and 1 is AI_GENERATED based on PRD logic
            # Confidence is the probability of the predicted class
            confidence = float(max(probabilities))
            label = "AI_GENERATED" if prediction == 1 else "HUMAN"
            
            return {
                "classification": label,
                "confidence": round(confidence, 2)
            }, None
        except Exception as e:
            return None, f"Inference error: {str(e)}"
