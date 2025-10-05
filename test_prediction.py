import pandas as pd
import numpy as np
from src.model import ExoplanetClassifier

def test_prediction():
    """Test the prediction with correct feature names"""
    # Initialize classifier
    classifier = ExoplanetClassifier()
    
    # Load model
    try:
        classifier.load_model('models/exoplanet_classifier_model.pkl')
        print("Model loaded successfully")
        print("Expected feature columns:", classifier.feature_columns)
    except Exception as e:
        print("Error loading model:", e)
        return
    
    # Create test data with correct feature names
    test_data = pd.DataFrame([{
        'koi_period': 15.2,
        'koi_time0bk': 2454968.3842,
        'koi_duration': 2.3,
        'koi_depth': 10.5,
        'koi_prad': 2.1,
        'koi_teq': 1200
    }])
    
    print("Test data:")
    print(test_data)
    print("Test data columns:", list(test_data.columns))
    
    try:
        # Make prediction
        predictions = classifier.predict(test_data)
        print("Predictions:", predictions)
    except Exception as e:
        print("Error during prediction:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()