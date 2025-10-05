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

class ExoplanetClassifier:
    """
    A class to handle exoplanet classification using machine learning models.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        self.best_model = None
        
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
    
    def train_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """
        Train multiple models and compare their performance
        """
        print("Training multiple models...")
        
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
        print(f"Best model: {best_model_name} with accuracy: {max([score for _, score in model_scores]):.4f}")
        
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
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
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
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # For neural network, save separately
        if 'NeuralNetwork' in str(type(self.best_model)):
            self.best_model.save(filepath)
            # Also save the preprocessing components
            joblib.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'imputer': self.imputer,
                'feature_columns': self.feature_columns
            }, filepath + '_preprocessing.pkl')
        else:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'imputer': self.imputer,
                'feature_columns': self.feature_columns
            }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        try:
            # Try loading as a neural network first
            nn_model = keras.models.load_model(filepath)
            preprocessing_data = joblib.load(filepath + '_preprocessing.pkl')
            self.best_model = nn_model
            self.scaler = preprocessing_data['scaler']
            self.label_encoder = preprocessing_data['label_encoder']
            self.imputer = preprocessing_data['imputer']
            self.feature_columns = preprocessing_data['feature_columns']
        except:
            # Load as a traditional model
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.imputer = model_data['imputer']
            self.feature_columns = model_data['feature_columns']

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

# Light curve analysis functions
def generate_synthetic_light_curve(period=5.0, duration=0.3, depth=0.02, snr=15.0, t_max=50.0):
    """
    Generate a synthetic light curve with a transit signal
    """
    # Create time array
    t = np.linspace(0, t_max, int(t_max * 24 * 2))  # 30-minute cadence
    
    # Initialize flux as 1.0 (normalized)
    flux = np.ones_like(t)
    
    # Add multiple transits based on period
    transit_times = np.arange(0, t_max, period)
    
    for transit_time in transit_times:
        # Create a transit shape (trapezoidal model for simplicity)
        in_transit = (t >= transit_time - duration/2) & (t <= transit_time + duration/2)
        
        if np.any(in_transit):
            # Create trapezoidal transit shape
            local_t = t[in_transit] - transit_time
            transit_shape = np.ones_like(local_t)
            
            # Simple trapezoidal shape
            ingress_egress = duration * 0.2  # ingress/egress duration
            
            # Ingress
            ingress_mask = (local_t >= -duration/2) & (local_t < -duration/2 + ingress_egress)
            transit_shape[ingress_mask] = 1 - depth + depth * (local_t[ingress_mask] + duration/2) / ingress_egress
            
            # Egress
            egress_mask = (local_t > duration/2 - ingress_egress) & (local_t <= duration/2)
            transit_shape[egress_mask] = 1 - depth + depth * (duration/2 - local_t[egress_mask]) / ingress_egress
            
            # Full transit (flat bottom)
            full_transit_mask = (local_t >= -duration/2 + ingress_egress) & (local_t <= duration/2 - ingress_egress)
            transit_shape[full_transit_mask] = 1 - depth
            
            flux[in_transit] = transit_shape
    
    # Add noise
    noise = np.random.normal(0, depth/snr, size=len(t))
    flux += noise
    
    return t, flux

def analyze_light_curve_features(time, flux):
    """
    Extract key features from the light curve for classification
    """
    features = {}
    
    try:
        # Convert to numpy arrays if they aren't already
        time = np.asarray(time)
        flux = np.asarray(flux)
        
        # Basic statistics with error handling
        features['mean_flux'] = float(np.mean(flux)) if len(flux) > 0 and not np.all(np.isnan(flux)) else 1.0
        features['std_flux'] = float(np.std(flux)) if len(flux) > 0 and not np.all(np.isnan(flux)) else 0.01
        features['median_flux'] = float(np.median(flux)) if len(flux) > 0 and not np.all(np.isnan(flux)) else 1.0
        features['min_flux'] = float(np.min(flux)) if len(flux) > 0 and not np.all(np.isnan(flux)) else 0.98
        features['max_flux'] = float(np.max(flux)) if len(flux) > 0 and not np.all(np.isnan(flux)) else 1.02
        features['flux_range'] = float(features['max_flux'] - features['min_flux']) if len(flux) > 0 else 0.04
        
        # Variability metrics
        features['variance'] = float(np.var(flux)) if len(flux) > 0 else 0.0001
        
        # Handle skewness and kurtosis with error handling
        try:
            import pandas as pd
            flux_series = pd.Series(flux)
            skew_val = flux_series.skew()
            kurt_val = flux_series.kurtosis()
            features['skewness'] = float(skew_val) if not (np.isnan(skew_val) or np.isinf(skew_val)) else 0.0
            features['kurtosis'] = float(kurt_val) if not (np.isnan(kurt_val) or np.isinf(kurt_val)) else 3.0
        except:
            features['skewness'] = 0.0
            features['kurtosis'] = 3.0
        
        # Periodicity analysis using periodogram with better error handling
        try:
            if len(time) > 20 and len(flux) > 20:  # Need sufficient data points
                # Calculate sampling interval
                dt_values = np.diff(time)
                dt_values = dt_values[~np.isnan(dt_values)]  # Remove NaN values
                dt_values = dt_values[np.isfinite(dt_values)]  # Remove infinite values
                
                if len(dt_values) > 0:
                    dt = np.median(dt_values)
                else:
                    dt = 1.0
                
                if dt > 0 and np.isfinite(dt) and dt != np.inf:
                    freqs = np.fft.fftfreq(len(time), dt)
                    fft_power = np.abs(np.fft.fft(flux))**2
                    positive_freqs = (freqs > 0) & np.isfinite(freqs) & np.isfinite(fft_power)
                    
                    if np.sum(positive_freqs) > 0:
                        dominant_freq_idx = np.argmax(fft_power[positive_freqs])
                        freq_val = freqs[positive_freqs][dominant_freq_idx]
                        if freq_val != 0 and np.isfinite(freq_val) and freq_val != np.inf:
                            period_val = 1.0 / freq_val
                            if np.isfinite(period_val) and period_val != np.inf:
                                features['dominant_period'] = float(abs(period_val))
                            else:
                                features['dominant_period'] = 5.0  # Default reasonable period
                        else:
                            features['dominant_period'] = 5.0
                    else:
                        features['dominant_period'] = 5.0  # Default period
                else:
                    features['dominant_period'] = 5.0
            else:
                features['dominant_period'] = 5.0  # Default period
        except Exception as e:
            features['dominant_period'] = 5.0  # Fallback if FFT fails
        
        # Transit-specific metrics with better error handling
        try:
            # Calculate rolling statistics to detect possible transits
            if len(flux) > 50:  # Need sufficient data points
                import pandas as pd
                flux_series = pd.Series(flux)
                rolling_window = min(50, len(flux) // 2)  # Adjust window size based on data length
                rolling_min = flux_series.rolling(window=rolling_window, center=True, min_periods=1).min()
                min_diff = flux_series - rolling_min
                
                # Find significant dips (potential transits)
                significant_dips = min_diff[(min_diff > 0) & np.isfinite(min_diff)]
                if len(significant_dips) > 0 and not np.all(np.isnan(significant_dips)):
                    max_drop = significant_dips.max()
                    avg_drop = significant_dips.mean()
                    if np.isfinite(max_drop) and max_drop != np.inf:
                        features['max_flux_drop'] = float(abs(max_drop))
                    else:
                        features['max_flux_drop'] = 0.02
                    
                    if np.isfinite(avg_drop) and avg_drop != np.inf:
                        features['avg_flux_drop'] = float(abs(avg_drop))
                    else:
                        features['avg_flux_drop'] = 0.01
                else:
                    features['max_flux_drop'] = 0.02  # Default transit depth
                    features['avg_flux_drop'] = 0.01
            else:
                features['max_flux_drop'] = 0.02
                features['avg_flux_drop'] = 0.01
        except Exception as e:
            features['max_flux_drop'] = 0.02
            features['avg_flux_drop'] = 0.01
        
        # Additional transit metrics
        try:
            # Find local minima (potential transit bottoms)
            from scipy.signal import find_peaks
            
            # Invert the flux to find minima as peaks
            inverted_flux = -flux
            # Ensure we have finite values
            inverted_flux = inverted_flux[np.isfinite(inverted_flux)]
            
            if len(inverted_flux) > 10:
                # Calculate threshold for peak detection
                mean_flux = np.mean(flux[np.isfinite(flux)]) if len(flux[np.isfinite(flux)]) > 0 else 1.0
                std_flux_val = np.std(flux[np.isfinite(flux)]) if len(flux[np.isfinite(flux)]) > 0 else 0.01
                
                peaks, _ = find_peaks(inverted_flux, height=-mean_flux + 0.1*std_flux_val)
                
                if len(peaks) > 0 and len(flux) > max(peaks):
                    # Calculate transit depths
                    valid_flux_indices = np.isfinite(flux)
                    if np.sum(valid_flux_indices) > 0:
                        mean_valid_flux = np.mean(flux[valid_flux_indices])
                        transit_depths = mean_valid_flux - flux[peaks]
                        transit_depths = transit_depths[np.isfinite(transit_depths)]
                        
                        if len(transit_depths) > 0:
                            mean_depth = np.mean(transit_depths)
                            max_depth = np.max(transit_depths)
                            
                            if np.isfinite(mean_depth) and mean_depth != np.inf:
                                features['mean_transit_depth'] = float(abs(mean_depth))
                            else:
                                features['mean_transit_depth'] = 0.02
                            
                            if np.isfinite(max_depth) and max_depth != np.inf:
                                features['max_transit_depth'] = float(abs(max_depth))
                            else:
                                features['max_transit_depth'] = 0.02
                        else:
                            features['mean_transit_depth'] = 0.02
                            features['max_transit_depth'] = 0.02
                    else:
                        features['mean_transit_depth'] = 0.02
                        features['max_transit_depth'] = 0.02
                    
                    # Calculate transit durations
                    if len(peaks) > 1 and len(time) > max(peaks):
                        valid_time_indices = np.isfinite(time)
                        if np.sum(valid_time_indices) > 1:
                            transit_intervals = np.diff(time[peaks])
                            transit_intervals = transit_intervals[np.isfinite(transit_intervals)]
                            
                            if len(transit_intervals) > 0:
                                mean_interval = np.mean(transit_intervals)
                                if np.isfinite(mean_interval) and mean_interval != np.inf:
                                    features['mean_transit_interval'] = float(abs(mean_interval))
                                else:
                                    features['mean_transit_interval'] = 5.0
                            else:
                                features['mean_transit_interval'] = 5.0
                        else:
                            features['mean_transit_interval'] = 5.0
                    else:
                        features['mean_transit_interval'] = 5.0
                else:
                    features['mean_transit_depth'] = 0.02
                    features['max_transit_depth'] = 0.02
                    features['mean_transit_interval'] = 5.0
            else:
                features['mean_transit_depth'] = 0.02
                features['max_transit_depth'] = 0.02
                features['mean_transit_interval'] = 5.0
        except Exception as e:
            features['mean_transit_depth'] = 0.02
            features['max_transit_depth'] = 0.02
            features['mean_transit_interval'] = 5.0
    
    except Exception as e:
        # If anything fails catastrophically, return reasonable defaults
        features = {
            'mean_flux': 1.0,
            'std_flux': 0.01,
            'median_flux': 1.0,
            'min_flux': 0.98,
            'max_flux': 1.02,
            'flux_range': 0.04,
            'variance': 0.0001,
            'skewness': 0.0,
            'kurtosis': 3.0,
            'dominant_period': 5.0,
            'max_flux_drop': 0.02,
            'avg_flux_drop': 0.01,
            'mean_transit_depth': 0.02,
            'max_transit_depth': 0.02,
            'mean_transit_interval': 5.0
        }
    
    # Ensure all features are valid numbers (no NaN or infinity)
    for key in list(features.keys()):
        try:
            # Convert to float and check for valid values
            val = float(features[key])
            if np.isnan(val) or np.isinf(val):
                features[key] = 0.0
            else:
                features[key] = val
        except (ValueError, TypeError):
            features[key] = 0.0
    
    # If all features are still zero or near-zero, set reasonable defaults
    if all(abs(v) < 1e-10 for v in features.values()):
        features = {
            'mean_flux': 1.0,
            'std_flux': 0.01,
            'median_flux': 1.0,
            'min_flux': 0.98,
            'max_flux': 1.02,
            'flux_range': 0.04,
            'variance': 0.0001,
            'skewness': 0.0,
            'kurtosis': 3.0,
            'dominant_period': 5.0,
            'max_flux_drop': 0.02,
            'avg_flux_drop': 0.01,
            'mean_transit_depth': 0.02,
            'max_transit_depth': 0.02,
            'mean_transit_interval': 5.0
        }
    
    return features

def predict_habitability(planet_params):
    """
    Predict habitability probability for a planet
    """
    try:
        # Extract parameters
        orbital_distance_au = float(planet_params.get('orbital_distance_au', 1.0))
        stellar_teff = float(planet_params.get('stellar_teff', 5778))
        stellar_radius = float(planet_params.get('stellar_radius', 1.0))
        equilibrium_temp = float(planet_params.get('equilibrium_temp', 288))
        radius_earth = float(planet_params.get('radius_earth', 1.0))
        
        # Calculate habitable zone boundaries using conservative estimates
        # Inner edge: runaway greenhouse effect
        inner_hz = stellar_radius**2 * (stellar_teff / 5778)**4 * 0.95
        
        # Outer edge: maximum greenhouse warming
        outer_hz = stellar_radius**2 * (stellar_teff / 5778)**4 * 1.4
        
        # Calculate scores for each factor
        # Distance score (0-1 scale, 1 = perfect position)
        if inner_hz <= orbital_distance_au <= outer_hz:
            distance_score = 1.0
        elif orbital_distance_au < inner_hz:
            # Too close - linear decrease from 1 to 0 as we approach 0
            distance_score = max(0.0, 1.0 - (inner_hz - orbital_distance_au) / inner_hz)
        else:
            # Too far - linear decrease from 1 to 0 as we move away
            distance_score = max(0.0, 1.0 - (orbital_distance_au - outer_hz) / outer_hz)
        
        # Temperature score (Earth-like temperatures are ideal: 200-300K)
        temp_deviation = abs(equilibrium_temp - 250)  # Earth's equilibrium temp ~250K
        temp_score = max(0.0, 1.0 - temp_deviation / 200.0)  # Linear decrease
        
        # Size score (Earth-sized planets are ideal: 0.5-2.0 Earth radii)
        if 0.5 <= radius_earth <= 2.0:
            size_score = 1.0
        elif radius_earth < 0.5:
            size_score = max(0.0, radius_earth / 0.5)  # Linear increase from 0 to 1
        else:
            size_score = max(0.0, 1.0 - (radius_earth - 2.0) / 2.0)  # Linear decrease from 1 to 0
        
        # Combined habitability probability (weighted average)
        habitability_prob = (
            0.4 * distance_score +  # Distance is most important (40%)
            0.3 * temp_score +     # Temperature is important (30%)
            0.3 * size_score       # Size is important (30%)
        )
        
        # Ensure probability is within valid range
        habitability_prob = max(0.0, min(1.0, habitability_prob))
        
        # Details dictionary with individual scores
        details = {
            'distance_score': float(distance_score),
            'temperature_score': float(temp_score),
            'size_score': float(size_score),
            'inner_hz': float(inner_hz),
            'outer_hz': float(outer_hz)
        }
        
        return float(habitability_prob), details
        
    except Exception as e:
        # Return default values if calculation fails
        return 0.0, {
            'distance_score': 0.0,
            'temperature_score': 0.0,
            'size_score': 0.0,
            'inner_hz': 0.95,
            'outer_hz': 1.4
        }

# Example usage
if __name__ == "__main__":
    classifier = ExoplanetClassifier()
    
    # Load data
    df = classifier.load_data('data/koi_data.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
    
    # Train models
    best_model_name = classifier.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Train neural network
    classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Print classification report for the best model
    y_pred = classifier.best_model.predict(X_test)
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, y_pred, target_names=classifier.label_encoder.classes_))
    
    # Save the model
    classifier.save_model('models/exoplanet_classifier_model.pkl')
    print("Model saved to models/exoplanet_classifier_model.pkl")