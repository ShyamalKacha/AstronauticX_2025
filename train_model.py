"""
Script to train the final optimized exoplanet classification model
"""
import pandas as pd
from src.model import ExoplanetClassifier
import os

def main():
    print("Starting ExoFinder model training...")
    
    # Create necessary directories
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Initialize classifier
    classifier = ExoplanetClassifier()
    
    # Load the Kepler dataset
    print("Loading Kepler dataset...")
    df = classifier.load_data('data/koi_data.csv')
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
    
    # Train multiple models
    print("Training multiple models...")
    best_model_name = classifier.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Train neural network
    print("Training neural network...")
    classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save the best performing model
    print(f"Saving the best model ({best_model_name})...")
    classifier.save_model('models/exoplanet_classifier_model.pkl')
    
    # Print model performance
    y_pred = classifier.best_model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print feature importance for the best model
    importance_df = classifier.get_feature_importance()
    if importance_df is not None:
        print("\nFeature Importance:")
        print(importance_df)
    
    print("\nExoFinder model training completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: models/exoplanet_classifier_model.pkl")

if __name__ == "__main__":
    main()