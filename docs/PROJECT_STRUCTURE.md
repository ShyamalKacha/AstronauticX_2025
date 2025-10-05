ExoFinder - Project Structure
=============================

├── data/                          # Dataset files
│   ├── koi_data.csv              # Kepler Objects of Interest dataset
│   ├── toi_data.csv              # TESS Objects of Interest (sample)
│   └── k2_data.csv               # K2 mission data (sample)
│
├── models/                        # Trained ML models
│   └── exoplanet_classifier_model.pkl
│
├── notebooks/                     # Jupyter notebooks for analysis
│   └── exoplanet_analysis.ipynb  # Dataset exploration and model design
│
├── src/                           # Source code
│   ├── model.py                  # ML model implementation
│   └── backend.py                # Flask backend API
│
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css             # Custom styling
│   ├── js/
│   │   └── main.js               # Client-side JavaScript
│   └── images/                   # Image assets
│
├── templates/                     # HTML templates
│   └── index.html                # Main web interface
│
├── docs/                          # Documentation
│   └── PROJECT_SUMMARY.md        # Project summary documentation
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main project documentation
└── download_datasets.py          # Dataset download script