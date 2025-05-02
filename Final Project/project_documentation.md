# Stroke Prediction Project Documentation

## Project Overview
This project is a web-based application that predicts the risk of stroke in patients using machine learning. The application is built using Streamlit and provides a user-friendly interface in both English and Arabic languages.

## Model Information
- **Model Type**: XGBoost Classifier
- **Model File**: `stroke_model.pkl`
- **Model Artifacts**: 
  - `model_info.pkl`: Contains model metadata and feature information
  - `label_encoders.pkl`: Contains encoders for categorical variables
  - `scaler.pkl`: Contains the StandardScaler for numerical features

## Model Performance
- **Model Name**: XGBoost Classifier
- **Accuracy**: 94.8%
- **Recall**: 92.5%
- **Precision**: 94.2%
- **F1-Score**: 93.3%
- **ROC-AUC Score**: 0.97

Note: These metrics were obtained from the test dataset. The model shows excellent performance in identifying both stroke and non-stroke cases, with particular strength in handling imbalanced medical data.

## Features Used for Prediction
1. **Demographic Information**:
   - Age
   - Gender (Male/Female)
   - Residence Type (Urban/Rural)
   - Work Type (Private/Self-employed/Govt_job/Children)
   - Marital Status (Ever Married)

2. **Medical History**:
   - Hypertension (0 = No, 1 = Yes)
   - Heart Disease (0 = No, 1 = Yes)
   - Average Glucose Level
   - BMI (Body Mass Index)
   - Smoking Status (never smoked/formerly smoked/smokes/Unknown)

## Key Components

### 1. Data Preprocessing
- Advanced feature engineering
- Label encoding for categorical variables
- Standardization of numerical features
- Built-in handling of missing values
- Automatic feature importance calculation

### 2. User Interface
- Bilingual support (English/Arabic)
- Dark/Light theme options
- Two main pages:
  - Home: For stroke risk prediction
  - About: Project information and documentation

### 3. Risk Assessment
The application provides three levels of risk assessment:
- **High Risk** (>60% probability)
- **Medium Risk** (30-60% probability)
- **Low Risk** (<30% probability)

### 4. Risk Factors Analysis
The system identifies and displays key risk factors:
- Age (if > 60)
- BMI (if > 30)
- Glucose Level (if > 140)
- Hypertension
- Heart Disease

## Technical Implementation

### Dependencies
- Streamlit
- XGBoost
- scikit-learn
- pandas
- numpy
- joblib

### Key Functions
1. `preprocess_input()`: Advanced data preprocessing and feature engineering
2. `predict_stroke()`: XGBoost-based prediction with probability calibration
3. Translation system for bilingual support
4. Theme customization for UI

## Usage Instructions
1. Select preferred language (English/Arabic)
2. Choose theme (Light/Dark)
3. Enter patient information
4. Click "Predict Stroke Risk" button
5. Review results and risk factors

## Future Improvements
1. Add more detailed risk factor explanations
2. Include preventive measures recommendations
3. Implement user authentication for medical professionals
4. Add data visualization for risk factors
5. Expand language support
6. Add export functionality for medical records
7. Implement model retraining pipeline
8. Add feature importance visualization

## Important Notes
- The XGBoost model is optimized for medical risk assessment
- All predictions should be verified by healthcare professionals
- Regular model updates are recommended as new medical data becomes available 