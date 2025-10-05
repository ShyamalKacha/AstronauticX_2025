import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
try:
    from .light_curve_analysis import generate_synthetic_light_curve, analyze_light_curve_features
except ImportError:
    from light_curve_analysis import generate_synthetic_light_curve, analyze_light_curve_features
try:
    from .habitability_calculator import calculate_habitability_probability
    from .light_curve_cnn import LightCurveCNN
except ImportError:
    from habitability_calculator import calculate_habitability_probability
    from light_curve_cnn import LightCurveCNN


class AdvancedExoplanetClassifier:
    """
    Advanced exoplanet classifier with multiple capabilities:
    - Traditional ML with tabular data
    - CNN with direct light curve analysis
    - Habitability assessment
    - Light curve feature extraction
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        self.best_model = None
        self.light_curve_cnn = None
        self.habitability_calculator = True
        
    def load_data(self, file_path):
        """
        Load the exoplanet dataset
        """
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {self.df.shape}")
        return self.df
    
    def prepare_data(self, df=None):
        """
        Prepare the data for training
        """
        if df is None:
            df = self.df
            
        # Prepare features and target variable
        X = df[self.feature_columns].copy()
        y = df['koi_disposition'].copy()
        
        # Handle missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features (only used for neural network and logistic regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """
        Train traditional ML models
        """
        print("Training traditional models...")
        
        # Initialize models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Train models that don't need scaling
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Train model that needs scaling
        lr_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        rf_pred = rf_model.predict(X_test)
        gb_pred = gb_model.predict(X_test)
        lr_pred = lr_model.predict(X_test_scaled)
        
        # Calculate accuracies
        rf_acc = accuracy_score(y_test, rf_pred)
        gb_acc = accuracy_score(y_test, gb_pred)
        lr_acc = accuracy_score(y_test, lr_pred)
        
        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
        print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
        
        # Store models
        self.models['RandomForest'] = rf_model
        self.models['GradientBoosting'] = gb_model
        self.models['LogisticRegression'] = lr_model
        
        # Create ensemble model
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lr', lr_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )
        
        # Train ensemble model with scaled data
        ensemble_model.fit(X_train_scaled, y_train)
        ensemble_pred = ensemble_model.predict(X_test_scaled)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"Ensemble Model Accuracy: {ensemble_acc:.4f}")
        
        # Store ensemble model
        self.models['Ensemble'] = ensemble_model
        
        # Determine best model
        model_scores = [('RandomForest', rf_acc), ('GradientBoosting', gb_acc), 
                       ('LogisticRegression', lr_acc), ('Ensemble', ensemble_acc)]
        
        best_model_name = max(model_scores, key=lambda x: x[1])[0]
        
        self.best_model = self.models[best_model_name]
        print(f"Best traditional model: {best_model_name} with accuracy: {max([score for _, score in model_scores]):.4f}")
        
        return best_model_name
    
    def train_neural_network(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """
        Train a neural network model
        """
        print("Training Neural Network...")
        
        # Encode labels for neural network
        from sklearn.preprocessing import LabelBinarizer
        label_binarizer = LabelBinarizer()
        y_train_nn = label_binarizer.fit_transform(y_train)
        y_test_nn = label_binarizer.transform(y_test)
        
        # Define and compile the neural network
        model_nn = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(y_train_nn.shape[1], activation='softmax')
        ])
        
        model_nn.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        # Train the neural network
        history = model_nn.fit(
            X_train_scaled, y_train_nn,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate the neural network
        test_loss, test_accuracy = model_nn.evaluate(X_test_scaled, y_test_nn, verbose=0)
        print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
        
        # Store neural network model
        self.models['NeuralNetwork'] = model_nn
        
        return model_nn, history
    
    def train_light_curve_cnn(self, n_samples=1000):
        """
        Train the light curve CNN model
        """
        print("Training Light Curve CNN...")
        
        # Generate training data
        from src.light_curve_cnn import generate_training_data
        light_curves, labels = generate_training_data(n_samples=n_samples)
        
        # Initialize CNN model
        self.light_curve_cnn = LightCurveCNN(sequence_length=500)
        
        # Train the model
        history = self.light_curve_cnn.train(light_curves, labels, epochs=20)
        
        print("Light Curve CNN training completed!")
        return self.light_curve_cnn, history
    
    def predict(self, X, method='best'):
        """
        Make predictions using specified method
        """
        if method == 'traditional' and self.best_model is not None:
            # Handle missing values
            X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
            
            if 'NeuralNetwork' in str(type(self.best_model)):
                # For neural network, we need scaled features
                X_scaled = self.scaler.transform(X_imputed)
                predictions = self.best_model.predict(X_scaled)
                # Convert probabilities to class predictions
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = self.best_model.predict(X_imputed)
        
            # Decode labels back to original names
            return self.label_encoder.inverse_transform(predictions)
        
        elif method == 'light_curve' and self.light_curve_cnn is not None:
            # Assuming X contains light curves
            predictions, probabilities = self.light_curve_cnn.predict(X)
            return predictions, probabilities
        
        elif method == 'best' and self.best_model is not None:
            # Use the best traditional model
            X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
            predictions = self.best_model.predict(X_imputed)
            return self.label_encoder.inverse_transform(predictions)
        else:
            raise ValueError("Model not trained or specified method not available")
    
    def predict_habitability(self, planet_params):
        """
        Predict habitability of an exoplanet
        """
        if self.habitability_calculator:
            habitability_prob, details = calculate_habitability_probability(planet_params)
            return habitability_prob, details
        else:
            return 0.0, {}
    
    def analyze_light_curve(self, time_series, flux):
        """
        Analyze a light curve and extract features
        """
        features = analyze_light_curve_features(time_series, flux)
        return features
    
    def generate_synthetic_light_curve(self, **params):
        """
        Generate a synthetic light curve with specified parameters
        """
        time, flux = generate_synthetic_light_curve(**params)
        return time, flux
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Save traditional models
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'models': self.models  # Save all models
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.imputer = model_data['imputer']
        self.feature_columns = model_data['feature_columns']
        self.models = model_data.get('models', {})

    def get_feature_importance(self):
        """
        Get feature importance if the model supports it
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None

# Example usage
if __name__ == "__main__":
    classifier = AdvancedExoplanetClassifier()
    
    # Demo light curve generation and analysis
    print("Demonstrating light curve analysis...")
    
    # Generate a synthetic light curve
    time, flux = classifier.generate_synthetic_light_curve(
        period=5.0, duration=0.3, depth=0.02, snr=15.0
    )
    
    # Analyze the light curve
    features = classifier.analyze_light_curve(time, flux)
    print("Extracted light curve features:")
    for key, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {key}: {value:.4f}")
    
    # Demo habitability calculation
    print("\nDemonstrating habitability calculation...")
    
    planet_params = {
        'orbital_distance_au': 1.0,  # 1 AU = Earth's distance
        'equilibrium_temp': 255,     # Earth's equilibrium temp without greenhouse
        'radius_earth': 1.0,         # Earth radius
        'stellar_teff': 5778,        # Sun's temperature in K
        'stellar_radius': 1.0        # Sun's radius in solar radii
    }
    
    prob, details = classifier.predict_habitability(planet_params)
    print(f"Habitability Probability: {prob:.3f}")
    print(f"Details: {details}")
    
    print("\nAdvanced Exoplanet Classifier initialized successfully!")
    print("Features available:")
    print("- Traditional ML models (Random Forest, Gradient Boosting, etc.)")
    print("- Light curve CNN for direct analysis")
    print("- Habitability assessment")
    print("- Light curve feature extraction and generation")