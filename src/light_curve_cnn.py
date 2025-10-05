import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
try:
    from .light_curve_analysis import generate_synthetic_light_curve, analyze_light_curve_features
except ImportError:
    from light_curve_analysis import generate_synthetic_light_curve, analyze_light_curve_features

class LightCurveCNN:
    """
    Convolutional Neural Network for analyzing raw light curve data directly
    """
    
    def __init__(self, sequence_length=500):
        self.sequence_length = sequence_length
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
    def build_model(self, n_classes=3):
        """
        Build the CNN model for light curve classification
        """
        model = keras.Sequential([
            # Input layer - reshape to (batch_size, sequence_length, 1)
            layers.Reshape((self.sequence_length, 1), input_shape=(self.sequence_length,)),
            
            # First convolutional block
            layers.Conv1D(64, 10, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv1D(128, 8, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv1D(256, 6, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Fourth convolutional block
            layers.Conv1D(512, 4, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Global Average Pooling instead of flattening to reduce parameters
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_light_curve_data(self, light_curves, labels):
        """
        Prepare light curves for CNN training
        light_curves: list of light curve arrays of variable length
        labels: corresponding labels
        """
        # Process each light curve to fixed length
        processed_curves = []
        
        for curve in light_curves:
            # If curve is too short, pad with the median value
            if len(curve) < self.sequence_length:
                median_val = np.median(curve)
                padded_curve = np.pad(curve, (0, self.sequence_length - len(curve)), 
                                     mode='constant', constant_values=median_val)
                processed_curves.append(padded_curve)
            # If curve is too long, truncate
            elif len(curve) > self.sequence_length:
                processed_curves.append(curve[:self.sequence_length])
            # If exactly right length
            else:
                processed_curves.append(curve)
        
        X = np.array(processed_curves)
        
        # Normalize the light curves
        X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
        
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
        else:
            y = self.label_encoder.transform(labels)
        
        return X, y
    
    def train(self, light_curves, labels, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the CNN model
        """
        if self.model is None:
            # Determine number of classes from labels
            unique_labels = list(set(labels))
            n_classes = len(unique_labels)
            self.build_model(n_classes)
        
        # Prepare the data
        X, y = self.prepare_light_curve_data(light_curves, labels)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, light_curves):
        """
        Make predictions on light curves
        """
        X, _ = self.prepare_light_curve_data(light_curves, ['dummy'] * len(light_curves))
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Decode labels back to original names
        if self.label_encoder:
            return self.label_encoder.inverse_transform(predicted_classes), predictions
        else:
            return predicted_classes, predictions
    
    def evaluate(self, light_curves, labels):
        """
        Evaluate the model
        """
        X, y = self.prepare_light_curve_data(light_curves, labels)
        test_loss, test_accuracy = self.model.evaluate(X, y, verbose=0)
        return test_loss, test_accuracy

def generate_training_data(n_samples=1000, use_real_data=True):
    """
    Generate training data for the light curve CNN using real NASA data when available
    """
    from process_nasa_data import prepare_real_nasa_data_for_cnn
    
    if use_real_data:
        print("Attempting to load real NASA exoplanet data...")
        real_data = prepare_real_nasa_data_for_cnn()
        
        if real_data:
            print("Using real NASA data for training")
            light_curves, labels = real_data
            # Pad or truncate to ensure consistent length
            processed_curves = []
            for curve in light_curves:
                if len(curve) > 500:
                    # Truncate
                    processed_curves.append(curve[:500])
                elif len(curve) < 500:
                    # Pad with median value
                    median_val = np.median(curve)
                    padded = np.pad(curve, (0, 500 - len(curve)), constant_values=median_val)
                    processed_curves.append(padded)
                else:
                    processed_curves.append(curve)
            return processed_curves, labels
        else:
            print("Real data not available, using synthetic data")
    
    # Fallback to synthetic data
    light_curves = []
    labels = []
    
    for i in range(n_samples):
        # Randomly decide if this is a transit signal or just noise
        if np.random.random() > 0.3:  # 70% chance of transit
            # Generate light curve with transit
            period = np.random.uniform(1, 30)  # Period between 1-30 days
            duration = np.random.uniform(0.1, 0.5)  # Duration between 0.1-0.5 days
            depth = np.random.uniform(0.001, 0.05)  # Depth between 0.001-0.05 (0.1%-5%)
            
            t, f = generate_synthetic_light_curve(
                period=period, 
                duration=duration, 
                depth=depth, 
                snr=np.random.uniform(5, 50)
            )
            
            light_curves.append(f)
            
            # Label based on whether it has a clear transit
            if depth > 0.01:  # Strong signal
                labels.append('CONFIRMED')
            else:  # Weaker signal
                labels.append('CANDIDATE')
        else:
            # Generate light curve with no transit (just noise)
            t = np.linspace(0, 50, 1000)
            f = np.ones_like(t) + np.random.normal(0, 0.005, size=len(t))
            light_curves.append(f)
            labels.append('FALSE POSITIVE')  # No transit = false positive
    
    return light_curves, labels

def demo_cnn_model():
    """
    Demonstrate the CNN model for light curve analysis
    """
    print("Generating training data...")
    light_curves, labels = generate_training_data(n_samples=2000)  # Smaller dataset for demo
    
    print(f"Generated {len(light_curves)} light curves")
    print("Sample labels:", np.unique(labels, return_counts=True))
    
    print("\nInitializing CNN model...")
    cnn_model = LightCurveCNN(sequence_length=500)  # Fixed length for CNN
    
    print("Training CNN model...")
    history = cnn_model.train(light_curves, labels, epochs=20)  # Fewer epochs for demo
    
    # Generate test data
    test_curves, test_labels = generate_training_data(n_samples=200)
    
    print("\nEvaluating model...")
    test_loss, test_accuracy = cnn_model.evaluate(test_curves, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Make predictions on a few examples
    sample_curves = test_curves[:5]
    predictions, probabilities = cnn_model.predict(sample_curves)
    
    print("\nSample Predictions:")
    for i, (pred, true_label, prob) in enumerate(zip(predictions, test_labels[:5], probabilities)):
        print(f"Sample {i+1}: Predicted={pred}, True={true_label}, Confidence={np.max(prob):.3f}")
    
    # Save the model
    cnn_model.model.save('models/light_curve_cnn_model.h5')
    print("\nCNN model saved to models/light_curve_cnn_model.h5")
    
    return cnn_model, history

if __name__ == "__main__":
    # Test the CNN model
    model, history = demo_cnn_model()