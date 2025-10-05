# ExoFinder - NASA Space App Challenge 2025

## Project Summary

ExoFinder is an AI-powered web application developed for the NASA Space App Challenge 2025. The project addresses the challenge "A World Away: Hunting for Exoplanets with AI" by creating a machine learning model that can analyze astronomical data to identify and classify exoplanets.

## Solution Overview

The solution combines:
- Machine learning models trained on NASA's Kepler Objects of Interest (KOI) dataset
- A web interface for user interaction
- Batch processing capabilities for analyzing multiple data points
- Ensemble methods for improved accuracy

## Technical Implementation

### 1. Data Processing
- Utilizes multiple NASA exoplanet datasets: Kepler Objects of Interest (KOI), TESS Objects of Interest (TOI), and K2 mission data
- Combined dataset includes over 8,000 samples from all missions
- Features include orbital period, transit duration, transit depth, planetary radius, and equilibrium temperature
- Implements data preprocessing with imputation for missing values
- Cross-dataset model training for improved generalization

### 2. Machine Learning Models
- **Random Forest**: For interpretability and handling non-linear relationships
- **Gradient Boosting**: For high accuracy in classification tasks
- **Logistic Regression**: As a baseline model
- **Neural Network**: For complex pattern recognition
- **Ensemble Method**: Combining multiple algorithms for robustness

### 3. Web Interface
- Responsive design using Bootstrap 5
- Single prediction form for manual input
- Batch processing via file upload
- Real-time classification results
- Informational sections about exoplanet detection

### 4. API Endpoints
- `GET /` - Main web interface
- `POST /api/train` - Train/retrain the ML model
- `POST /api/predict` - Predict single exoplanet classification
- `POST /api/predict_from_file` - Batch prediction from CSV
- `GET /api/model_info` - Get model metadata
- `POST /api/upload_data` - Upload new dataset

## Key Features

1. **Multi-Model Approach**: Implements several ML algorithms and selects the best performer
2. **Ensemble Techniques**: Combines multiple models for improved accuracy
3. **Web Accessibility**: User-friendly interface for researchers and enthusiasts
4. **Batch Processing**: Supports CSV file uploads for bulk analysis
5. **Real-time Predictions**: Instant classification of astronomical parameters

## Data Preprocessing

The pipeline includes:
- Missing value imputation using median values
- Feature scaling for neural networks and logistic regression
- Label encoding for categorical target variables
- Train/test split with stratification

## Model Selection

Based on analysis of the Kepler dataset, the ensemble approach typically shows the best performance:
- Random Forest: ~75% accuracy
- Gradient Boosting: ~74% accuracy
- Logistic Regression: ~69% accuracy
- Neural Network: Variable accuracy depending on tuning

## Technology Stack

- **Backend**: Python Flask
- **ML Libraries**: Scikit-learn, TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **API**: Flask RESTful endpoints

## Installation & Deployment

1. Clone the repository
2. Create Python virtual environment
3. Install dependencies from requirements.txt
4. Download and place Kepler dataset in /data directory
5. Run `python src/backend.py`
6. Access application at http://localhost:5000

## Results & Impact

This solution demonstrates how AI can accelerate the discovery of exoplanets by:
- Reducing manual analysis time
- Identifying potential candidates that might be missed by human analysis
- Providing a scalable approach to classify large datasets
- Making exoplanet research more accessible to the public

## Future Enhancements

Potential improvements include:
- Integration with TESS and K2 datasets
- Time-series analysis with CNNs/LSTMs directly on light curves
- Advanced feature engineering from raw photometric data
- Real-time data streaming capabilities
- Integration with additional astronomical databases

## Conclusion

ExoFinder successfully addresses the NASA Space App Challenge by combining machine learning with space science to accelerate exoplanet discovery. The web interface makes this sophisticated analysis accessible to researchers and enthusiasts alike, contributing to our understanding of planetary systems beyond our solar system.