import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test the light curve analysis functions
import numpy as np
from src.light_curve_analysis import analyze_light_curve_features

def test_light_curve_analysis():
    """Test the light curve analysis function with sample data"""
    print("Testing light curve analysis function...")
    
    # Sample data similar to what would come from the frontend
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    flux = np.array([0.999, 0.998, 0.997, 0.996, 0.995, 0.996, 0.997, 0.998, 0.999, 0.998, 0.997])
    
    print(f"Input time array: {time}")
    print(f"Input flux array: {flux}")
    
    # Analyze the light curve
    features = analyze_light_curve_features(time, flux)
    
    print("\nExtracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Check if we have meaningful values (not all zeros)
    non_zero_features = {k: v for k, v in features.items() if abs(v) > 1e-10}
    print(f"\nNon-zero features: {len(non_zero_features)} out of {len(features)}")
    
    if len(non_zero_features) > 0:
        print("SUCCESS: Light curve analysis working correctly")
        print("Sample non-zero features:")
        for k, v in list(non_zero_features.items())[:5]:  # Show first 5
            print(f"  {k}: {v}")
    else:
        print("ERROR: All features are zero - there may be an issue")

if __name__ == "__main__":
    test_light_curve_analysis()