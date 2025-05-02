import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

def load_data(file_path='../data/train_v1.csv'):
    """Load and perform initial data cleaning"""
    print("Loading data...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print("Initial shape:", df.shape)
        
        # Handle missing values strategically
        # For BMI, create groups and fill with group medians
        df['bmi_group'] = pd.qcut(df['bmi'].dropna(), q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['age_group'] = pd.qcut(df['age'], q=5, labels=['very_young', 'young', 'middle', 'old', 'very_old'])
        
        # Fill BMI missing values based on age and gender groups
        for gender in df['gender'].unique():
            for age_group in df['age_group'].unique():
                mask = (df['gender'] == gender) & (df['age_group'] == age_group) & (df['bmi'].isna())
                median_bmi = df[(df['gender'] == gender) & (df['age_group'] == age_group)]['bmi'].median()
                df.loc[mask, 'bmi'] = median_bmi
        
        # Fill remaining BMI missing values with overall median
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        
        # For smoking_status, create a more informed 'Unknown' category
        df['smoking_status'].fillna('Unknown', inplace=True)
        
        # Drop temporary columns
        df = df.drop(['bmi_group', 'age_group'], axis=1)
        
        print("Final shape:", df.shape)
        print("\nMissing values after cleaning:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the data and create feature encoders"""
    print("\nPreprocessing data...")
    
    # Create copies of encoders for later use
    label_encoders = {}
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Apply label encoding to categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"\nEncoding mapping for {col}:")
        for i, label in enumerate(le.classes_):
            print(f"{label}: {i}")
    
    # Create more meaningful features
    df['age_risk'] = np.where(df['age'] > 60, 1, 0)  # Age risk factor
    df['bmi_risk'] = np.where(df['bmi'] > 30, 1, 0)  # BMI risk factor
    df['glucose_risk'] = np.where(df['avg_glucose_level'] > 140, 1, 0)  # Glucose risk factor
    df['total_risk_factors'] = df['hypertension'] + df['heart_disease'] + df['age_risk'] + df['bmi_risk'] + df['glucose_risk']
    
    # Scale numerical features using RobustScaler
    scaler = RobustScaler()
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save preprocessors
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Prepare features and target
    X = df.drop(['stroke', 'id'], axis=1)
    y = df['stroke']
    
    # Apply SMOTE with adjusted parameters
    print("\nApplying SMOTE to balance the dataset...")
    print("Original class distribution:")
    print(pd.Series(y).value_counts(normalize=True))
    
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Reduced sampling ratio
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("\nBalanced class distribution:")
    print(pd.Series(y_resampled).value_counts(normalize=True))
    
    return X_resampled, y_resampled, label_encoders, scaler

def train_and_evaluate_models(X, y):
    """Train multiple models and select the best one"""
    print("\nTraining and evaluating models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models with optimized parameters
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=10,      # Reduced from 15
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=5,    # Increased from 2
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=5,  # Added class weight
            random_state=42,
            n_jobs=-1
        ),
        'logistic': LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    all_results = []
    
    # Train and evaluate each model with cross-validation
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        results = {
            'model': name,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_recall_mean': cv_scores.mean(),
            'cv_recall_std': cv_scores.std()
        }
        all_results.append(results)
        
        print(f"{name} Results:")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"CV Recall Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Update best model if current is better
        if recall > best_score:
            best_score = recall
            best_model = model
            best_model_name = name
    
    print("\nAll models results:")
    results_df = pd.DataFrame(all_results)
    print(results_df[['model', 'recall', 'precision', 'f1', 'roc_auc', 'cv_recall_mean', 'cv_recall_std']])
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best recall score: {best_score:.4f}")
    
    return best_model, X_test, y_test

def plot_feature_importance(model, X, output_file='feature_importance.png'):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print("\nTop 10 most important features:")
        print(importance.head(10))

def save_model_artifacts(model, X, label_encoders, scaler):
    """Save model and preprocessing artifacts"""
    print("\nSaving model artifacts...")
    joblib.dump(model, 'stroke_model.pkl')
    
    # Save feature names
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    # Save model info
    model_info = {
        'feature_names': X.columns.tolist(),
        'categorical_cols': list(label_encoders.keys()),
        'numerical_cols': ['age', 'avg_glucose_level', 'bmi']
    }
    joblib.dump(model_info, 'model_info.pkl')

def main():
    # Load and preprocess data
    df = load_data()
    X, y, label_encoders, scaler = preprocess_data(df)
    
    # Train and evaluate models
    best_model, X_test, y_test = train_and_evaluate_models(X, y)
    
    # Generate feature importance plot
    plot_feature_importance(best_model, X)
    
    # Save model and artifacts
    save_model_artifacts(best_model, X, label_encoders, scaler)
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    print("\nFinal Model Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main()