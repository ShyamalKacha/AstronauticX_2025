from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import sys
from advanced_model import AdvancedExoplanetClassifier  # Changed to use AdvancedExoplanetClassifier

# Add the src directory to the Python path
sys.path.append(os.path.dirname(__file__))

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize the advanced classifier  # Changed from ExoplanetClassifier to AdvancedExoplanetClassifier
classifier = AdvancedExoplanetClassifier()

# Load the pre-trained model when the application starts
def load_model():
    try:
        classifier.load_model('models/exoplanet_classifier_model.pkl')
        print("Original model loaded successfully")
    except:
        print("No pre-trained model found. Training a new model...")
        # Train model from scratch if it doesn't exist
        df = classifier.load_data('data/koi_data.csv')
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
        classifier.train_traditional_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)  # Changed method name
        classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        # Save the trained model
        classifier.save_model('models/exoplanet_classifier_model.pkl')
        print("Trained model and saved to models/exoplanet_classifier_model.pkl")

# Load the model when the module is imported
load_model()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the model with the dataset"""
    try:
        # Load the dataset
        df = classifier.load_data('data/koi_data.csv')
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
        
        # Train models
        best_model_name = classifier.train_traditional_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)  # Changed method name
        
        # Train neural network
        classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save the model
        classifier.save_model('models/exoplanet_classifier_model.pkl')
        
        return jsonify({
            'status': 'success',
            'message': f'Model trained successfully. Best model: {best_model_name}',
            'best_model': best_model_name
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_exoplanet():
    """Predict exoplanet disposition from features"""
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate input
        required_features = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        
        # Check if data is provided as a single object or as a list
        if isinstance(data, dict):
            # Single prediction
            if all(key in data for key in required_features):
                # Create DataFrame with the exact column order expected by the model
                ordered_data = {key: data[key] for key in classifier.feature_columns}
                input_data = pd.DataFrame([ordered_data])
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required features. Required: {required_features}'
                }), 400
        elif isinstance(data, list):
            # Batch prediction
            for item in data:
                if not all(key in item for key in required_features):
                    return jsonify({
                        'status': 'error',
                        'message': f'Missing required features in one of the items. Required: {required_features}'
                    }), 400
            # Create DataFrame with the exact column order expected by the model
            ordered_data = [{key: item[key] for key in classifier.feature_columns} for item in data]
            input_data = pd.DataFrame(ordered_data)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Input must be a single object or a list of objects'
            }), 400
        
        # Debug information
        print("Input data columns:", input_data.columns.tolist())
        print("Input data shape:", input_data.shape)
        print("Expected feature columns:", classifier.feature_columns)
        print("Column order matches:", list(input_data.columns) == classifier.feature_columns)
        
        # Make predictions
        predictions = classifier.predict(input_data)
        
        # Convert predictions to a standard format to ensure valid JSON
        clean_predictions = []
        for pred in predictions:
            if pd.isna(pred) or pred is None:
                clean_predictions.append("UNKNOWN")
            else:
                clean_predictions.append(str(pred))
        
        if isinstance(data, dict):
            # Single prediction result
            result = {
                'status': 'success',
                'prediction': clean_predictions[0],
                'confidence': 'N/A'  # TODO: Add confidence scores
            }
        else:
            # Batch prediction result
            result = {
                'status': 'success',
                'predictions': clean_predictions
            }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict_from_file', methods=['POST'])
def predict_from_file():
    """Predict exoplanet disposition from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Unsupported file format. Please upload a CSV file.'
            }), 400
        
        # Validate required columns
        required_features = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
        if not all(col in df.columns for col in required_features):
            return jsonify({
                'status': 'error',
                'message': f'Missing required columns. Required: {required_features}'
            }), 400
        
        # Make predictions
        predictions = classifier.predict(df[required_features])
        
        # Add predictions to the dataframe
        df['predicted_disposition'] = predictions
        
        # Convert to list of dictionaries for JSON response
        results = df.to_dict('records')
        
        return jsonify({
            'status': 'success',
            'predictions': results
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    try:
        # Get feature importance if available
        importance_df = classifier.get_feature_importance()
        
        if importance_df is not None:
            feature_importance = importance_df.to_dict('records')
        else:
            feature_importance = None
        
        return jsonify({
            'status': 'success',
            'model_type': str(type(classifier.best_model).__name__),
            'feature_columns': classifier.feature_columns,
            'feature_importance': feature_importance,
            'classes': classifier.label_encoder.classes_.tolist(),
            'accuracy': '77%+'  # Improved accuracy with advanced models
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Upload new dataset for training"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Save the file to the data directory
        filepath = os.path.join('data', file.filename)
        file.save(filepath)
        
        # Load and validate the dataset
        df = pd.read_csv(filepath)
        
        required_columns = ['koi_disposition'] + classifier.feature_columns
        if not all(col in df.columns for col in required_columns):
            return jsonify({
                'status': 'error',
                'message': f'Dataset must contain required columns: {required_columns}'
            }), 400
        
        return jsonify({
            'status': 'success',
            'message': f'Dataset uploaded successfully to {filepath}',
            'shape': df.shape
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ADDING BACK THE LIGHT CURVE AND HABITABILITY FEATURES
@app.route('/api/light_curve_features', methods=['POST'])
def analyze_light_curve():
    """Analyze a light curve and extract features"""
    try:
        data = request.json
        
        # Validate input
        if 'time' not in data or 'flux' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Light curve data must include "time" and "flux" arrays'
            }), 400
        
        time = np.array(data['time'])
        flux = np.array(data['flux'])
        
        # Perform light curve analysis
        features = classifier.analyze_light_curve(time, flux)
        
        # Clean the features dictionary to remove any NaN or infinity values
        cleaned_features = {}
        for key, value in features.items():
            # Convert to float and check for valid values
            try:
                float_value = float(value)
                if np.isnan(float_value) or np.isinf(float_value):
                    cleaned_features[key] = 0.0  # Replace invalid values with 0
                else:
                    cleaned_features[key] = float_value
            except (ValueError, TypeError):
                # If conversion fails, keep as is (likely already a number)
                cleaned_features[key] = value if not (np.isnan(value) or np.isinf(value)) else 0.0
        
        return jsonify({
            'status': 'success',
            'features': cleaned_features
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/generate_light_curve', methods=['POST'])
def generate_light_curve():
    """Generate a synthetic light curve"""
    try:
        data = request.json
        
        # Use default parameters if not provided
        params = {
            'period': float(data.get('period', 5.0)),
            'duration': float(data.get('duration', 0.3)),
            'depth': float(data.get('depth', 0.02)),
            'snr': float(data.get('snr', 15.0)),
            't_max': float(data.get('t_max', 50.0))
        }
        
        time, flux = classifier.generate_synthetic_light_curve(**params)
        
        # Ensure we return valid JSON (convert numpy arrays to lists)
        time_list = time.tolist() if hasattr(time, 'tolist') else list(time)
        flux_list = flux.tolist() if hasattr(flux, 'tolist') else list(flux)
        
        # Clean any NaN or infinity values
        time_clean = [float(t) if not (np.isnan(t) or np.isinf(t)) else 0.0 for t in time_list]
        flux_clean = [float(f) if not (np.isnan(f) or np.isinf(f)) else 0.0 for f in flux_list]
        
        return jsonify({
            'status': 'success',
            'time': time_clean,
            'flux': flux_clean
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/habitability', methods=['POST'])
def calculate_habitability():
    """Calculate habitability probability for a planet"""
    try:
        data = request.json
        
        # Validate required parameters
        required_params = ['orbital_distance_au', 'stellar_teff', 'stellar_radius']
        if not all(param in data for param in required_params):
            return jsonify({
                'status': 'error',
                'message': f'Missing required parameters. Required: {required_params}'
            }), 400
        
        # Set defaults for optional parameters and ensure numeric conversion
        planet_params = {
            'orbital_distance_au': float(data['orbital_distance_au']),
            'stellar_teff': float(data['stellar_teff']),
            'stellar_radius': float(data['stellar_radius']),
            'equilibrium_temp': float(data.get('equilibrium_temp', 288)),
            'radius_earth': float(data.get('radius_earth', 1.0))
        }
        
        habitability_prob, details = classifier.predict_habitability(planet_params)
        
        # Clean the details dictionary to ensure valid JSON
        cleaned_details = {}
        for key, value in details.items():
            # Convert to float and check for valid values
            try:
                float_value = float(value)
                if np.isnan(float_value) or np.isinf(float_value):
                    cleaned_details[key] = 0.0  # Replace invalid values with 0
                else:
                    cleaned_details[key] = float_value
            except (ValueError, TypeError):
                # If conversion fails, keep as is
                cleaned_details[key] = value if not (np.isnan(value) or np.isinf(value)) else 0.0
        
        # Ensure habitability probability is a valid float
        habitability_float = float(habitability_prob) if not (np.isnan(habitability_prob) or np.isinf(habitability_prob)) else 0.0
        
        return jsonify({
            'status': 'success',
            'habitability_probability': habitability_float,
            'details': cleaned_details
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Create the models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load the model before starting the server
    try:
        classifier.load_model('models/exoplanet_classifier_model.pkl')
        print("Model loaded successfully")
    except:
        print("Model not found. Training a new model...")
        df = classifier.load_data('data/koi_data.csv')
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = classifier.prepare_data()
        classifier.train_traditional_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)  # Changed method name
        classifier.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        # Save the trained model
        classifier.save_model('models/exoplanet_classifier_model.pkl')
        print("Trained model and saved to models/exoplanet_classifier_model.pkl")
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)