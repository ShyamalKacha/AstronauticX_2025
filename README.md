# ExoFinder Pro - AI-Powered Exoplanet Discovery Platform

**AstronauticX Team - NASA Space App Challenge 2025**

## 🚀 Project Overview

ExoFinder Pro is an advanced AI-powered exoplanet discovery and classification system designed for NASA's Space App Challenge. Our platform uses machine learning algorithms to analyze exoplanet data from space missions like Kepler, K2, and TESS to identify and classify potential exoplanets in the search for habitable worlds.

## 🌟 Key Features

- **Multi-Dataset Analysis**: Works with Kepler, K2, and TESS data for comprehensive exoplanet detection
- **Light Curve Analysis**: Advanced algorithms to detect subtle transit signals in stellar brightness data
- **Real-time Classification**: Instant classification of exoplanet candidates using trained ML models
- **Habitability Assessment**: Probability calculation for potentially habitable conditions
- **Web Interface**: Intuitive dashboard for space researchers and enthusiasts
- **Batch Processing**: Analyze multiple candidates at once for efficiency
- **Feature Visualization**: Graphical representation of light curves and key parameters
- **Direct Light Curve CNN**: Uses convolutional neural networks to analyze raw light curve data without extensive pre-processing

## 📊 Datasets

The application is trained on multiple NASA exoplanet datasets from different space missions:

### Primary Dataset
- **Kepler Objects of Interest (KOI)**: Light curve data from Kepler space telescope with classifications of objects as confirmed planets, candidates, or false positives

### Additional Datasets
- **TESS Objects of Interest (TOI)**: Data from the Transiting Exoplanet Survey Satellite
- **K2 Mission Data**: Extended mission data from the Kepler space telescope

All datasets include astronomical parameters like orbital period, transit duration, transit depth, planetary radius, and equilibrium temperature.

## 🏗️ Technical Architecture

- **Backend**: Python Flask API
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **Frontend**: HTML, CSS (Bootstrap), JavaScript
- **Models**: Random Forest, Gradient Boosting, Logistic Regression, Neural Network, Ensemble, CNN for direct light curve analysis

### Core Models:
1. **Random Forest** - For interpretability and accuracy
2. **Gradient Boosting** - For high accuracy
3. **Logistic Regression** - For baseline performance
4. **Neural Network** - For complex pattern recognition
5. **Ensemble Method** - Combining all approaches for robustness
6. **Light Curve CNN** - For direct analysis of raw light curve data

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Navigate to the project directory:
```bash
cd ExoFinder-Pro
```

3. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Prepare the data directory:
```bash
mkdir -p data models
```

6. Run the application:
```bash
python src/backend.py
```

7. Access the application at `http://localhost:5000`

## 🎯 Usage

### Web Interface
1. Access the main dashboard at `http://localhost:5000`
2. Navigate to different sections using the top navigation bar:
   - **Home**: Overview and quick access
   - **Classify**: Single and batch prediction
   - **Data**: Dataset information and statistics
   - **Light Curve**: Analyze and generate light curves
   - **Habitability**: Calculate habitability probabilities
   - **About**: Project information

### Single Prediction
1. Go to the "Classify" section
2. Enter astronomical parameters:
   - Orbital Period (days)
   - Transit Epoch (BJD)
   - Transit Duration (hours)
   - Transit Depth (ppm)
   - Planetary Radius (R_Earth)
   - Equilibrium Temperature (K)
3. Click "Classify Exoplanet"

### Batch Analysis
1. Prepare your CSV file with required columns
2. Go to the "Classify" section
3. Upload your CSV file in the batch analysis section
4. View the batch analysis results

### Light Curve Analysis
1. Go to the "Light Curve" section
2. Enter time and flux arrays (comma-separated) or use the generator
3. Analyze the extracted features

### Habitability Assessment
1. Go to the "Habitability" section
2. Enter planetary parameters:
   - Orbital Distance (AU)
   - Stellar Temperature (K)
   - Stellar Radius (R_Sun)
   - Equilibrium Temperature (K)
   - Planetary Radius (R_Earth)
3. Get the habitability probability and analysis

## 🌐 API Endpoints

### Main Interface
- `GET /` - Main web interface

### Prediction Endpoints
- `POST /api/predict` - Single exoplanet prediction
- `POST /api/predict_from_file` - Batch prediction from CSV file

### Light Curve Endpoints
- `POST /api/light_curve_features` - Analyze light curve and extract features
- `POST /api/generate_light_curve` - Generate synthetic light curve

### Habitability Endpoints
- `POST /api/habitability` - Calculate habitability probability

### Model Information
- `GET /api/model_info` - Get information about the trained model
- `POST /api/upload_data` - Upload new dataset for training

## 🔬 How It Works

ExoFinder Pro uses multiple machine learning models to analyze exoplanet data:

1. **Data Processing**: The system processes light curve data from NASA's exoplanet missions, identifying periodic dips in star brightness that indicate potential planetary transits.

2. **Feature Extraction**: Key features such as orbital period, transit duration, transit depth, planetary radius, and equilibrium temperature are extracted from the light curves.

3. **Classification**: Advanced ensemble models (Random Forest, Gradient Boosting, Neural Networks) classify candidates as CONFIRMED, CANDIDATE, or FALSE POSITIVE.

4. **Habitability Assessment**: The system evaluates the potential habitability of discovered planets using multiple factors including orbital distance, temperature, and planetary size.

## 🌍 Habitability Assessment

The habitability calculator uses multiple factors to determine the likelihood of habitable conditions:
- **Distance Score**: How close the planet is to the habitable zone
- **Temperature Score**: Whether the planet's temperature allows for liquid water
- **Size Score**: Whether the planet's size is Earth-like and suitable for life

The habitable zone is calculated using conservative and optimistic models based on stellar parameters.

## 📈 Model Performance

The ensemble approach typically achieves high accuracy on the Kepler dataset. The model can classify astronomical signals into three categories:
- **CONFIRMED**: Verified exoplanets
- **CANDIDATE**: Potential exoplanets requiring verification
- **FALSE POSITIVE**: Non-exoplanet signals

## 🔧 Technologies Used

- **Python 3.8+**
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Neural networks
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Bootstrap 5** - Frontend framework
- **JavaScript** - Client-side functionality
- **Chart.js** - Interactive data visualization

## 📁 Project Structure

```
ExoFinder-Pro/
│
├── assessment/           # Model assessment and evaluation
├── data/                 # Dataset files
├── docs/                 # Documentation
├── models/               # Trained ML models
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── advanced_model.py  # Main classifier class
│   ├── backend.py         # Flask web application
│   ├── habitability_calculator.py  # Habitability calculations
│   ├── light_curve_analysis.py     # Light curve processing
│   ├── light_curve_cnn.py         # CNN for light curve analysis
│   └── process_nasa_data.py       # Data processing utilities
├── static/               # Static assets
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── images/           # Images
├── templates/            # HTML templates
│   └── index.html        # Main page
├── .gitignore            # Git ignore file
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── final_demo.py         # Final demonstration script
└── train_model.py        # Model training script
```

## 🌟 Innovation

ExoFinder Pro introduces several innovative approaches to exoplanet discovery:

- **Hybrid Ensemble Models**: Combines multiple ML algorithms for improved accuracy and reliability
- **Direct Light Curve CNN**: Uses convolutional neural networks to analyze raw light curve data without extensive pre-processing
- **Multi-Dataset Integration**: Seamlessly combines data from different space missions for comprehensive analysis
- **Real-time Habitability Prediction**: Advanced algorithms that consider multiple factors for habitability assessment
- **Adaptive Feature Engineering**: Automatically generates relevant features from raw data to improve model performance
- **Uncertainty Quantification**: Provides confidence scores for all predictions to assist researchers in decision-making

## 📊 Value Proposition

ExoFinder Pro provides significant value to the space research community:

- **Accelerated Discovery**: Dramatically reduces the time required to identify and classify exoplanet candidates
- **Improved Accuracy**: Ensemble models provide more accurate classifications than single algorithms
- **Reduced Manual Effort**: Automates the initial triage of candidates, allowing scientists to focus on promising discoveries
- **Accessibility**: Makes exoplanet research more accessible to amateur astronomers and educational institutions
- **Scientific Insights**: Helps identify potentially habitable worlds for future space missions and study
- **Data Democratization**: Provides public access to sophisticated exoplanet analysis tools
- **Research Acceleration**: Enables faster advancement in the search for Earth-like planets and life beyond our solar system

## 📚 NASA Data Sources

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Kepler Objects of Interest (KOI) dataset
- TESS Objects of Interest (TOI) dataset
- K2 mission dataset

## 🧪 Testing and Validation

The project includes several test files:
- `test_light_curve.py` - Light curve analysis tests
- `test_light_curve_features.py` - Feature extraction tests
- `test_prediction.py` - Prediction accuracy tests

Run tests with:
```bash
python -m pytest tests/
```

## 🚀 Running the Demo

To run a complete demonstration of the system's capabilities:
```bash
python final_demo.py
```

This will show:
- Model loading and initialization
- Data processing from NASA datasets
- Light curve analysis and feature extraction
- Habitability calculations
- Multi-disciplinary approach demonstration
- Integration of real NASA datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes (if applicable)
5. Run tests to ensure everything works
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is developed for the NASA Space App Challenge and is intended for educational and research purposes.

## 📞 Contact

AstronauticX Team - NASA Space App Challenge 2025

## 🏆 Competition Details

This project was developed for the NASA Space App Challenge 2025, focusing on:
- Innovative use of NASA data
- Scientific accuracy and relevance
- Real-world applicability
- User-friendly interface
- Educational value

## 🛑 Known Issues

- The model requires training data to be in a specific format (CSV with required columns)
- Large batch processing may require additional memory
- Habitability calculations are approximate and based on simplified models

## ⚡ Performance Notes

- The neural network model may take several minutes to train initially
- Light curve analysis can be computationally intensive for long time series
- The web interface is optimized for modern browsers

## 📖 Additional Resources

- [NASA Exoplanet Archive Documentation](https://exoplanetarchive.ipac.caltech.edu/docs/)
- [Kepler Mission Science Data](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [TESS Mission Science Data](https://www.nasa.gov/mission_pages/tess/main/index.html)
- [Exoplanet Research Overview](https://exoplanets.nasa.gov/)