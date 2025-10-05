import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

# Add the src directory to the Python path
sys.path.append(os.path.dirname(__file__))

def assess_exoplanet_model():
    """
    Assess the exoplanet classification model with comprehensive metrics
    """
    print("Starting model assessment...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/koi_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['koi_disposition'].value_counts()}")
    
    # Define features and target
    feature_columns = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
    X = df[feature_columns].copy()
    y = df['koi_disposition'].copy()
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    
    print(f"Classes: {classes}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features (for models that need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train models
    print("Training models...")
    
    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Train models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    lr_pred = lr_model.predict(X_test_scaled)
    
    # Also get prediction probabilities for ROC curves (for multiclass)
    rf_pred_proba = rf_model.predict_proba(X_test)
    gb_pred_proba = gb_model.predict_proba(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)
    
    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr', lr_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble_model.predict(X_test_scaled)
    ensemble_pred_proba = ensemble_model.predict_proba(X_test_scaled)
    
    # Calculate accuracies
    rf_acc = accuracy_score(y_test, rf_pred)
    gb_acc = accuracy_score(y_test, gb_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Ensemble Model Accuracy: {ensemble_acc:.4f}")
    
    # Select best model for detailed assessment
    models = {
        'RandomForest': (rf_model, rf_pred, rf_pred_proba),
        'GradientBoosting': (gb_model, gb_pred, gb_pred_proba),
        'LogisticRegression': (lr_model, lr_pred, lr_pred_proba),
        'Ensemble': (ensemble_model, ensemble_pred, ensemble_pred_proba)
    }
    
    best_model_name = max([('RandomForest', rf_acc), ('GradientBoosting', gb_acc), 
                          ('LogisticRegression', lr_acc), ('Ensemble', ensemble_acc)], 
                         key=lambda x: x[1])[0]
    
    best_model, best_pred, best_pred_proba = models[best_model_name]
    
    print(f"\nBest model for detailed assessment: {best_model_name}")
    
    # Create assessment folder if it doesn't exist
    os.makedirs('assessment', exist_ok=True)
    
    # Generate assessment matrices and save them
    generate_assessment_matrices(y_test, best_pred, best_pred_proba, classes, best_model_name, 
                                feature_columns, best_model, X_test, X_test_scaled, scaler, imputer)
    
    print("Model assessment completed. Results saved in 'assessment' folder.")


def generate_assessment_matrices(y_test, y_pred, y_pred_proba, classes, model_name, 
                                feature_columns, model, X_test, X_test_scaled, scaler, imputer):
    """
    Generate various assessment matrices and visualizations
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'assessment/confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate precision, recall, F1 for each class
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Add overall metrics
    overall_precision = precision_score(y_test, y_pred, average='weighted')
    overall_recall = recall_score(y_test, y_pred, average='weighted')
    overall_f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics report
    with open(f'assessment/metrics_report_{model_name.lower()}.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Overall Precision (weighted): {overall_precision:.4f}\n")
        f.write(f"Overall Recall (weighted): {overall_recall:.4f}\n")
        f.write(f"Overall F1-Score (weighted): {overall_f1:.4f}\n\n")
        f.write("Per-class metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write(f"\n\nDetailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=classes))
    
    # 2. ROC Curves for multiclass (One-vs-Rest approach)
    # For multiclass, we'll create ROC curves for each class
    n_classes = len(classes)
    
    # Binarize the labels for ROC calculation
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    # Plot ROC curves for each class
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i in range(min(n_classes, 3)):  # Only plot up to 3 classes
        if y_pred_proba.shape[1] > i:  # Check if we have prediction probabilities for this class
            # Get the prediction probabilities for the current class
            y_pred_proba_class = y_pred_proba[:, i]
            
            # Calculate ROC curve and AUC for the current class
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_class)
            auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba_class)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{classes[i]} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'assessment/roc_curves_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # If model is ensemble, we need to handle it differently
        # For voting classifier trained on scaled data, we'll use the random forest importances
        if model_name == 'Ensemble' or 'Voting' in str(type(model)):
            # Use random forest for feature importances if available
            rf_model = [est for name, est in model.named_estimators_.items() if 'RandomForest' in str(type(est))]
            if rf_model:
                importances = rf_model[0].feature_importances_
            else:
                # Use another model that supports feature importance
                importances = [est for name, est in model.named_estimators_.items()][0].feature_importances_
        
        # Create feature importance plot
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'assessment/feature_importance_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Classification Report Visualization
    # Create a heatmap of the classification report metrics
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    # Convert to DataFrame for visualization
    report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.3f', 
                cbar_kws={'label': 'Score'})
    plt.title(f'Classification Report Heatmap - {model_name}')
    plt.tight_layout()
    plt.savefig(f'assessment/classification_report_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Precision-Recall Curves (for multiclass)
    plt.figure(figsize=(10, 8))
    
    for i in range(min(n_classes, 3)):
        if y_pred_proba.shape[1] > i:
            # Get precision and recall for the current class
            precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(recall_vals, precision_vals, color=colors[i], lw=2, 
                    label=f'{classes[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'assessment/precision_recall_curves_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Assessment matrices saved for {model_name} model")


def run_cross_dataset_assessment():
    """
    Run assessment on cross-dataset model if available
    """
    print("\nRunning cross-dataset assessment...")
    
    # Check if we have TESS or K2 data to create a combined assessment
    datasets = []
    
    if os.path.exists('data/koi_data.csv'):
        koi_df = pd.read_csv('data/koi_data.csv')
        print(f"Kepler dataset: {len(koi_df)} samples")
        datasets.append(('Kepler', koi_df))
    
    if os.path.exists('data/toi_for_model.csv'):
        try:
            tess_df = pd.read_csv('data/toi_for_model.csv')
            # Rename columns to match Kepler naming convention
            column_mapping = {
                'toi_period': 'koi_period',
                'toi_duration': 'koi_duration',
                'toi_depth': 'koi_depth',
                'toi_prad': 'koi_prad',
                'toi_teq': 'koi_teq',
                'tfopwg_disp': 'koi_disposition',
                'disposition': 'koi_disposition'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in tess_df.columns:
                    tess_df = tess_df.rename(columns={old_col: new_col})
            
            print(f"TESS dataset: {len(tess_df)} samples")
            datasets.append(('TESS', tess_df))
        except:
            print("TESS dataset not available in the expected format")
    
    if os.path.exists('data/k2_for_model.csv'):
        try:
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
            
            print(f"K2 dataset: {len(k2_df)} samples")
            datasets.append(('K2', k2_df))
        except:
            print("K2 dataset not available in the expected format")
    
    # If we have multiple datasets, create a combined assessment
    if len(datasets) > 1:
        print("Combining datasets for cross-dataset assessment...")
        
        # Find common columns across all datasets
        common_cols = set(datasets[0][1].columns)
        for name, df in datasets[1:]:
            common_cols = common_cols & set(df.columns)
        
        # Only include datasets that have the required columns
        required_cols = {'koi_disposition', 'koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq'}
        if required_cols.issubset(common_cols):
            combined_df = None
            for name, df in datasets:
                if combined_df is None:
                    combined_df = df.copy()
                else:
                    # Align to the same columns
                    df_aligned = df[list(common_cols)]
                    combined_df = pd.concat([combined_df, df_aligned], ignore_index=True)
                    print(f"Added {name} data: {len(df_aligned)} additional samples")
            
            if combined_df is not None:
                print(f"Total combined dataset: {len(combined_df)} samples")
                
                # Perform cross-dataset assessment
                feature_columns = ['koi_period', 'koi_time0bk', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
                
                # Use only rows that have all required columns and target
                combined_df = combined_df.dropna(subset=feature_columns + ['koi_disposition'])
                
                if len(combined_df) > 0:
                    print(f"Dataset after cleaning: {len(combined_df)} samples")
                    
                    # Train model on combined dataset and assess
                    X = combined_df[feature_columns].copy()
                    y = combined_df['koi_disposition'].copy()
                    
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    classes = label_encoder.classes_
                    
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train a model on combined data
                    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    
                    # Make predictions
                    rf_pred = rf_model.predict(X_test)
                    rf_pred_proba = rf_model.predict_proba(X_test)
                    
                    # Generate assessment matrices
                    generate_assessment_matrices(y_test, rf_pred, rf_pred_proba, classes, 
                                              "Cross_Dataset_RF", feature_columns, 
                                              rf_model, X_test, X_test_scaled, scaler, imputer)
                
        else:
            print("Not enough common columns for cross-dataset assessment")
    else:
        print("Not enough datasets for cross-dataset assessment")


if __name__ == "__main__":
    assess_exoplanet_model()
    run_cross_dataset_assessment()