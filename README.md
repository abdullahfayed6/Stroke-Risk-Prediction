# Stroke Risk Prediction System

A machine learning-based web application that predicts the risk of stroke in patients using various health parameters. The application provides a user-friendly interface in both English and Arabic languages.

## Features

- 🧠 Advanced XGBoost-based stroke risk prediction
- 🌐 Bilingual support (English/Arabic)
- 🎨 Dark/Light theme options
- 📊 Detailed risk factor analysis
- 💻 User-friendly web interface
- 📱 Responsive design
- 🔄 Built-in handling of missing values
- 📈 Automatic feature importance calculation

## Project Structure

```
stroke-prediction/
├── app/
│   └── app.py               # Streamlit application
├── models/                  # Trained models and artifacts
├── static/                  # Static assets
├── data/                    # Dataset
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── project_documentation.md # Workflow and design details
└── model_comparison.md      # ML models comparison
```

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Streamlit application:
```bash
streamlit run app/app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8501
```

## Model Information

The system uses an XGBoost Classifier with the following performance metrics:
- Accuracy: 94.8%
- Recall: 92.5%
- Precision: 94.2%
- F1-Score: 93.3%
- ROC-AUC Score: 0.97

Key Features:
- Optimized for medical risk assessment
- Excellent handling of imbalanced data
- Built-in feature importance analysis
- Robust to missing values

For detailed model comparison and technical information, see [Model Comparison](model_comparison.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
