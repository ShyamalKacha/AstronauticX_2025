#!/usr/bin/env python
# Test script to check light curve feature extraction

from src.advanced_model import AdvancedExoplanetClassifier

def test_light_curve_features():
    print("Testing light curve feature extraction...")
    
    classifier = AdvancedExoplanetClassifier()

    # Generate a synthetic light curve
    time, flux = classifier.generate_synthetic_light_curve(
        period=5.0, 
        duration=0.3, 
        depth=0.02, 
        snr=15.0
    )
    
    print(f"Generated light curve with {len(time)} points")
    print(f"Time range: {time.min():.2f} to {time.max():.2f}")
    print(f"Flux range: {flux.min():.6f} to {flux.max():.6f}")

    # Analyze the light curve
    features = classifier.analyze_light_curve(time, flux)
    
    print('\nExtracted features:')
    for key, value in features.items():
        print(f'  {key}: {value}')
    
    # Check for expected features
    expected_features = ['mean_transit_depth', 'max_transit_depth', 'mean_transit_interval', 'dominant_period']
    print(f'\nChecking for expected features...')
    for feature in expected_features:
        if feature in features:
            print(f'  [OK] {feature}: {features[feature]}')
        else:
            print(f'  [MISSING] {feature}: NOT FOUND')

    # Check if features are 0
    zero_features = {k: v for k, v in features.items() if v == 0.0}
    if zero_features:
        print(f'\nZero-value features ({len(zero_features)}):')
        for k, v in zero_features.items():
            print(f'  {k}: {v}')
    else:
        print('\nNo zero-value features found.')
        
    print(f'\nFeatures found: {len(features)}')
    print('First 10 features:')
    for i, (k, v) in enumerate(features.items()):
        if i < 10:
            print(f'  {k}: {v}')

if __name__ == "__main__":
    test_light_curve_features()