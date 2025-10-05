"""
Script to demonstrate multi-dataset training for ExoFinder
This script shows how to train the model on different datasets (Kepler, TESS, K2)
"""
import pandas as pd
import numpy as np
from src.model import ExoplanetClassifier
import os

def prepare_cross_dataset_model():
    """
    Prepare a model that can work with all three datasets (Kepler, TESS, K2)
    """
    print("Preparing cross-dataset exoplanet classifier...")
    
    classifier = ExoplanetClassifier()
    
    # Load all datasets
    datasets = {}
    
    # Load Kepler dataset (primary dataset with most data)
    if os.path.exists('data/koi_data.csv'):
        print("Loading Kepler dataset...")
        kepler_df = pd.read_csv('data/koi_data.csv')
        datasets['Kepler'] = kepler_df
        print(f"Kepler dataset: {len(kepler_df)} samples")
    
    # Load TESS dataset
    if os.path.exists('data/toi_for_model.csv'):
        print("Loading TESS dataset...")
        tess_df = pd.read_csv('data/toi_for_model.csv')
        # Rename columns to match Kepler naming convention where possible
        column_mapping = {
            'toi_period': 'koi_period',
            'toi_time0bk': 'koi_time0bk', 
            'toi_duration': 'koi_duration',
            'toi_depth': 'koi_depth',
            'toi_prad': 'koi_prad',
            'toi_teq': 'koi_teq',
            'tfopwg_disp': 'koi_disposition',  # Common disposition column
            'disposition': 'koi_disposition'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in tess_df.columns:
                tess_df = tess_df.rename(columns={old_col: new_col})
        
        datasets['TESS'] = tess_df
        print(f"TESS dataset: {len(tess_df)} samples")
    
    # Load K2 dataset
    if os.path.exists('data/k2_for_model.csv'):
        print("Loading K2 dataset...")
        k2_df = pd.read_csv('data/k2_for_model.csv')
        # Rename columns to match Kepler naming convention
        column_mapping = {
            'period': 'koi_period',
            'duration': 'koi_duration',
            'depth': 'koi_depth',
            'k2_disp': 'koi_disposition',
            'disposition': 'koi_disposition'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in k2_df.columns:
                k2_df = k2_df.rename(columns={old_col: new_col})
        
        datasets['K2'] = k2_df
        print(f"K2 dataset: {len(k2_df)} samples")
    
    # Combine all datasets if they exist
    combined_df = None
    for name, df in datasets.items():
        if combined_df is None:
            combined_df = df.copy()
        else:
            # Only combine if they have the same columns
            common_cols = set(combined_df.columns) & set(df.columns)
            if 'koi_disposition' in common_cols and len(common_cols) >= 4:  # At least target + 3 features
                # Align columns
                df_aligned = df[list(common_cols)]
                combined_df = pd.concat([combined_df, df_aligned], ignore_index=True)
                print(f"Combined {name} data: {len(df_aligned)} additional samples")
    
    if combined_df is not None:
        print(f"\nCombined dataset: {len(combined_df)} total samples")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Show disposition distribution
        if 'koi_disposition' in combined_df.columns:
            print("\nDisposition distribution:")
            print(combined_df['koi_disposition'].value_counts())
        
        # Train the model on combined data
        classifier.df = combined_df
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
        
        best_model_name = classifier.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        
        # Train neural network as well
        classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save the trained model
        classifier.save_model('models/cross_dataset_exoplanet_classifier.pkl')
        print(f"\nCross-dataset model saved to models/cross_dataset_exoplanet_classifier.pkl")
        print(f"Best model: {best_model_name}")
        
        # Show feature importance
        importance_df = classifier.get_feature_importance()
        if importance_df is not None:
            print("\nFeature Importance:")
            print(importance_df)
        
        return classifier
    else:
        print("No datasets could be loaded for training")
        return None

def test_model_on_all_datasets():
    """
    Test the trained model on all available datasets
    """
    print("\nTesting model on all available datasets...")
    
    try:
        classifier = ExoplanetClassifier()
        classifier.load_model('models/cross_dataset_exoplanet_classifier.pkl')
        print("Loaded cross-dataset model successfully")
    except:
        print("Could not load cross-dataset model, trying original model...")
        try:
            classifier = ExoplanetClassifier()
            classifier.load_model('models/exoplanet_classifier_model.pkl')
            print("Loaded original model successfully")
        except:
            print("No trained model found, using untrained classifier")
            classifier = ExoplanetClassifier()
            # Train on Kepler data as fallback
            if os.path.exists('data/koi_data.csv'):
                df = pd.read_csv('data/koi_data.csv')
                classifier.df = df
                X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
                classifier.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
                classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
                classifier.save_model('models/exoplanet_classifier_model.pkl')
    
    # Test predictions on sample data
    print("\nSample predictions:")
    
    # Kepler-like sample
    kepler_sample = pd.DataFrame({
        'koi_period': [3.5, 15.2, 45.7],
        'koi_time0bk': [2454968.3842, 2455123.4567, 2455289.1234],
        'koi_duration': [2.3, 3.1, 4.8],
        'koi_depth': [10.5, 8.2, 15.3],
        'koi_prad': [1.2, 2.5, 4.1],
        'koi_teq': [1200, 800, 600]
    })
    
    try:
        predictions = classifier.predict(kepler_sample)
        print("Kepler-like predictions:", predictions)
    except Exception as e:
        print(f"Error predicting Kepler-like data: {e}")
    
    # TESS-like sample (using available columns)
    tess_sample = pd.DataFrame({
        'koi_period': [2.1, 10.5, 30.2],
        'koi_duration': [1.8, 2.9, 4.1],
        'koi_depth': [8.3, 12.7, 20.1],
        'koi_prad': [1.5, 3.2, 5.8],  # Assuming this maps correctly
        'koi_teq': [1400, 900, 700],
        'koi_time0bk': [2457000.5, 2457100.3, 2457200.1]
    })
    
    try:
        predictions = classifier.predict(tess_sample)
        print("TESS-like predictions:", predictions)
    except Exception as e:
        print(f"Error predicting TESS-like data: {e}")
    
    # K2-like sample
    k2_sample = pd.DataFrame({
        'koi_period': [5.2, 20.1, 55.6],
        'koi_duration': [2.1, 3.7, 5.2],
        'koi_depth': [12.4, 6.8, 18.9],
        'koi_prad': [2.1, 1.8, 6.3],
        'koi_teq': [1100, 750, 500],
        'koi_time0bk': [2456500.2, 2456700.8, 2456900.5]
    })
    
    try:
        predictions = classifier.predict(k2_sample)
        print("K2-like predictions:", predictions)
    except Exception as e:
        print(f"Error predicting K2-like data: {e}")

def main():
    print("ExoFinder - Multi-Dataset Training")
    print("="*50)
    
    # Train on combined datasets
    classifier = prepare_cross_dataset_model()
    
    # Test the model
    test_model_on_all_datasets()
    
    print("\n" + "="*50)
    print("Cross-dataset training and testing completed!")
    print("The model can now work with Kepler, TESS, and K2 data formats.")
    print("="*50)

if __name__ == "__main__":
    main()