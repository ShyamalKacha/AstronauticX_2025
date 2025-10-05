# ExoFinder Model Assessment Results

This folder contains comprehensive evaluations of the ExoFinder exoplanet classification models. The assessment includes various performance metrics, visualizations, and model comparisons.

## Files Overview

### Text Reports
- `metrics_report_randomforest.txt` - Detailed performance metrics for the best-performing Random Forest model
- `model_comparison.txt` - Comparative analysis of all models (Random Forest, Gradient Boosting, Logistic Regression)
- `summary_report.txt` - Overall summary of model performance and findings

### Visualizations
- `confusion_matrix_randomforest.png` - Confusion matrix showing classification accuracy per class
- `roc_curves_randomforest.png` - ROC curves for each class with AUC scores
- `precision_recall_curves_randomforest.png` - Precision-recall curves for each class
- `feature_importance_randomforest.png` - Feature importance ranking for Random Forest model
- `classification_report_randomforest.png` - Heatmap visualization of classification metrics
- `model_comparison.png` - Bar chart comparing all models' performance metrics

## Key Findings

### Model Performance
- **Best Model**: Random Forest (75.85% accuracy)
- **Second Best**: Gradient Boosting (74.43% accuracy)  
- **Third**: Logistic Regression (68.90% accuracy)

### Class-Specific Performance (Random Forest)
- **CANDIDATE**: Lower performance (37% F1-score, 29% recall) - hardest to classify
- **CONFIRMED**: Good performance (78% F1-score, 84% recall) - well-classified
- **FALSE POSITIVE**: Best performance (85% F1-score, 87% recall) - most distinguishable

### Feature Importance
The most important features for exoplanet classification are likely:
- Transit depth
- Planetary radius
- Orbital period
- (Specific order shown in feature_importance_randomforest.png)

## Model Strengths
- High accuracy in identifying false positives (85% F1-score)
- Good performance on confirmed exoplanets (78% F1-score)
- Overall weighted F1-score of 74.32%

## Model Limitations
- Struggles with candidate exoplanets (37% F1-score)
- Lower precision on candidate class (51%)
- Class imbalance may affect performance on minority class

## Recommendations
- Collect more representative training data for candidate exoplanets
- Consider ensemble methods or hyperparameter tuning for better performance
- Focus on features that better distinguish candidates from other classes