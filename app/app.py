import streamlit as st 
import joblib
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load model and artifacts at startup
model_path = Path("../models/stroke_model.pkl")
model_info_path = Path("../models/model_info.pkl")
label_encoders_path = Path("../models/label_encoders.pkl")
scaler_path = Path("../models/scaler.pkl")

try:
    model = joblib.load(model_path)
    model_info = joblib.load(model_info_path)
    label_encoders = joblib.load(label_encoders_path)
    scaler = joblib.load(scaler_path)
    print("Successfully loaded all model artifacts")
except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")
    st.stop()

print("Loaded model features:", model.feature_names_in_)

language = st.sidebar.radio("Select Language | اختر اللغة", ["English", "العربية"])
translations = {
      
    "English": {
        "title": "Welcome to our System",
        "input_prompt": "Enter Patient Details to Predict Stroke Risk",
        "age": "Age",
        "hypertension": "Hypertension (0 = No, 1 = Yes)",
        "heart_disease": "Heart Disease (0 = No, 1 = Yes)",
        "bmi": "BMI",
        "glucose": "Average Glucose Level",
        "smoking_status": "Smoking Status",
        "predict_button": "Predict Stroke Risk",
        "about_title": "About This App",
        "about_description": "This application predicts the risk of stroke based on patient data...",
        "project_overview": "Project Overview",
        "how_it_works": "How It Works",
        "theme_label": "Select Theme",
        "home": "Home",
        "about": "About",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "age": "Age",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "ever_married": "Ever Married",
        "no": "No",
        "yes": "Yes",
        "work_type": "Work Type",
        "private": "Private",
        "self_employed": "Self-employed",
        "govt_job": "Govt_job",
        "children": "children",
        "residence_type": "Residence Type",
        "urban": "Urban",
        "rural": "Rural",
        "avg_glucose_level": "Average Glucose Level",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "bmi": "BMI",
        "smoking_status": "Smoking Status",
        "never_smoked": "never smoked",
        "formerly_smoked": "formerly smoked",
        "smokes": "smokes",
        "unknown": "Unknown",
        "why_it_matters": "Why It Matters",
        "select_theme": "Select theme",
        "predict_stroke_risk": "Predict Stroke Risk",
        "light_mode": "Light Mode",
        "dark_mode": "Dark Mode"
    },


    "العربية": {
        "title": "مرحبًا بك في نظامنا",
        "input_prompt": "أدخل تفاصيل المريض للتنبؤ بمخاطر السكتة الدماغية",
        "age": "العمر",
        "hypertension": "ارتفاع ضغط الدم (0 = لا, 1 = نعم)",
        "heart_disease": "أمراض القلب (0 = لا, 1 = نعم)",
        "bmi": "مؤشر كتلة الجسم",
        "glucose": "متوسط مستوى الجلوكوز",
         "smoking_status": "حالة التدخين",
        "predict_button": "توقع خطر السكتة الدماغية",
        "about_title": "عن هذا التطبيق",
        "about_description": "يقوم هذا التطبيق بتوقع خطر الإصابة بالسكتة الدماغية بناءً على بيانات المريض...",
        "project_overview": "نظرة عامة على المشروع",
        "how_it_works": "كيف يعمل",
        "theme_label": "اختر الوضع",
        "home": "الرئيسية",
        "about": "عن التطبيق",
        "gender": "الجنس",
        "male": "ذكر",
        "female": "أنثى",
        "age": "العمر",
        "hypertension": "ارتفاع ضغط الدم",
        "heart_disease": "أمراض القلب",
        "ever_married": "متزوج مسبقًا",
        "no": "لا",
        "yes": "نعم",
        "work_type": "نوع العمل",
        "private": "قطاع خاص",
        "self_employed": "عمل حر",
        "govt_job": "وظيفة حكومية",
        "children": "طالب/طفل",
        "residence_type": "نوع السكن",
        "urban": "مدني",
        "rural": "ريفي",
        "avg_glucose_level": "متوسط مستوى الجلوكوز",
        "weight": "الوزن (كجم)",
        "height": "الطول (سم)",
        "bmi": "مؤشر كتلة الجسم",
        "smoking_status": "حالة التدخين",
        "never_smoked": "لم يدخن أبدًا",
        "formerly_smoked": "كان مدخنًا سابقًا",
        "smokes": "يدخن",
        "unknown": "غير معروف",
        "why_it_matters": "لماذا هذا مهم",
        "select_theme": "اختر النمط",
        "predict_stroke_risk": "توقع خطر السكتة الدماغية",
        "light_mode": "الوضع الفاتح",
        "dark_mode": "الوضع الداكن"

    }
}

theme = st.selectbox(
    translations[language]["select_theme"], 
    [translations[language]["light_mode"], translations[language]["dark_mode"]])


if theme == translations[language]["dark_mode"]:
    st.image("../static/S.png" , width=400)
    st.markdown(
        """
        <style>
        .stApp { background-color: #1E1E1E; color: white; }
        .stSidebar { background-color: #2E2E2E; }
        .stButton>button { background-color: #4CAF50; color: white; }
        label, .stRadio label, .stSelectbox label, .stSlider label, p, h1, h2, h3, h4, h5, h6 { color: white !important; }
        </style>
        """,
        unsafe_allow_html=True)
    
else:
    st.image("../static/S.png",width=400)
    st.markdown(
        """
        <style>
        .stApp { background-color: #FFFFFF; color: black; }
        .stSidebar { background-color: #F0F0F0; }
        .stButton>button { background-color: #008CBA; color: white; }
        label, .stRadio label, .stSelectbox label, .stSlider label, p, h1, h2, h3, h4, h5, h6 { color: black !important; }
        </style>
        """,
        unsafe_allow_html=True)
    

page = st.sidebar.radio("Navigation", [translations[language]["home"], translations[language]["about"]])
    
if page == translations[language]["home"]:
    
    st.markdown(f'<h1 class="center-text">{translations[language]["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="center-text">{translations[language]["input_prompt"]}</p>', unsafe_allow_html=True)

    # Gender selection 
    gender = st.radio(
        translations[language]["gender"],
        [translations[language]["male"], translations[language]["female"]],
        format_func=lambda x: f" {x}" if x == translations[language]["male"] else f"{x}"
    )

    # Age input
    age = st.number_input(f" {translations[language]['age']}", min_value=0, max_value=100, value=30)

    # Medical conditions 
    
    hypertension = st.radio(
        f"{translations[language]['hypertension']}", 
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    heart_disease = st.radio(
        f" {translations[language]['heart_disease']}", 
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    # Marital status 
    ever_married = st.radio(
        f"{translations[language]['ever_married']}", 
        [translations[language]["no"], translations[language]["yes"]]
    )

    # Work type
    work_options = {
        translations[language]["private"]: "Private Sector Employee",
        translations[language]["self_employed"]: "Self Employed / Business Owner",
        translations[language]["govt_job"]: "Government Employee",
        translations[language]["children"]: " Student/Child"
    }
    
    work_type = st.selectbox(
        f"{translations[language]['work_type']}", 
        list(work_options.keys()),
        format_func=lambda x: work_options[x]
    )

    # Residence type 
    residence = st.radio(
        f" {translations[language]['residence_type']}", 
        [translations[language]["urban"], translations[language]["rural"]],
        format_func=lambda x: f" {x}" if x == translations[language]["urban"] else f"{x}"
    )

    # Health metrics 
    glucose = st.number_input(f" {translations[language]['avg_glucose_level']}", min_value=0.0, max_value=300.0, value=100.0)
    weight = st.number_input(f"{translations[language]['weight']}", min_value=1.0, max_value=200.0, value=70.0)
    height = st.number_input(f"{translations[language]['height']}", min_value=50.0, max_value=250.0, value=170.0)

    bmi = weight / ((height / 100) ** 2)
    
    # BMI display with theme-responsive styling
    if theme == translations[language]["dark_mode"]:
        st.markdown(f"""
        <div style='background-color: #2E2E2E; padding: 15px; border-radius: 10px; margin: 10px 0; color: white;'>
            <h4>{translations[language]['bmi']}: {bmi:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; color: #2c3e50;'>
            <h4>{translations[language]['bmi']}: {bmi:.2f}</h4>
        </div>
        """, unsafe_allow_html=True)

    # Smoking status 
    smoking_options = {
        translations[language]["never_smoked"]: "Never Smoked",
        translations[language]["formerly_smoked"]: "Formerly Smoked",
        translations[language]["smokes"]: "Currently Smoking",
        translations[language]["unknown"]: " Unknown"
    }
    
    smoking_status = st.selectbox(
        f"{translations[language]['smoking_status']}", 
        list(smoking_options.keys()),
        format_func=lambda x: smoking_options[x]
    )

    # Style the predict button
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            height: 3em;
            font-size: 1.2em;
            font-weight: bold;
            border-radius: 10px;
            margin-top: 20px;
            background-color: #2196F3;
            color: white;
        }
        .stButton>button:hover {
            background-color: #1976D2;
        }
        </style>
    """, unsafe_allow_html=True)

    def preprocess_input(data):
        """Preprocess input data to match the trained model's expectations"""
        # Create DataFrame with correct column names
        df = pd.DataFrame([data], columns=["gender", "age", "hypertension", "heart_disease", "ever_married",
                                         "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"])
        
        # Create risk factor features
        df['age_risk'] = np.where(df['age'] > 60, 1, 0)
        df['bmi_risk'] = np.where(df['bmi'] > 30, 1, 0)
        df['glucose_risk'] = np.where(df['avg_glucose_level'] > 140, 1, 0)
        df['total_risk_factors'] = df['hypertension'] + df['heart_disease'] + df['age_risk'] + df['bmi_risk'] + df['glucose_risk']
        
        # Define the exact categories used during training
        category_mapping = {
            'gender': {
                translations[language]["male"]: "Male",
                translations[language]["female"]: "Female"
            },
            'ever_married': {
                translations[language]["yes"]: "Yes",
                translations[language]["no"]: "No"
            },
            'work_type': {
                translations[language]["private"]: "Private",
                translations[language]["self_employed"]: "Self-employed",
                translations[language]["govt_job"]: "Govt_job",
                translations[language]["children"]: "children"
            },
            'Residence_type': {
                translations[language]["urban"]: "Urban",
                translations[language]["rural"]: "Rural"
            },
            'smoking_status': {
                translations[language]["never_smoked"]: "never smoked",
                translations[language]["formerly_smoked"]: "formerly smoked",
                translations[language]["smokes"]: "smokes",
                translations[language]["unknown"]: "Unknown"
            }
        }
        
        # Map the input values to the exact categories used during training
        for col, mapping in category_mapping.items():
            df[col] = df[col].map(mapping)
        
        # Apply label encoding to categorical features
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_features:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])
        
        # Scale numerical features
        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        df[numerical_features] = df[numerical_features].astype(float)
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Ensure all required features are present
        required_features = model_info['feature_names']
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return df[required_features]


    def predict_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
        """Predict stroke risk using the trained model"""
        try:
            # Create input data with proper translations
            input_data = {
                "gender": gender,
                "age": float(age),
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": float(avg_glucose_level),
                "bmi": float(bmi),
                "smoking_status": smoking_status
            }
            
            # Preprocess the input data
            processed_data = preprocess_input(input_data)
            
            # Make prediction
            probability = model.predict_proba(processed_data)[0][1]
            
            # Determine risk level
            if probability > 0.6:
                risk_level = "High Risk"
            elif probability > 0.3:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            return probability, risk_level
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

    if st.button(translations[language]["predict_button"]):
        try:
            probability, risk_level = predict_stroke(
                gender, age, hypertension, heart_disease, ever_married,
                work_type, residence, glucose, bmi, smoking_status
            )
            
            if probability is not None:
                # Create styled risk level display
                if risk_level == "High Risk":
                    st.markdown(f"""
                    <div style='background-color: #ff4444; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                        <h2>{risk_level}</h2>
                        <h3>Stroke Probability: {probability:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium Risk":
                    st.markdown(f"""
                    <div style='background-color: #ffbb33; padding: 20px; border-radius: 10px; color: black; text-align: center;'>
                        <h2>{risk_level}</h2>
                        <h3>Stroke Probability: {probability:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #00C851; padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                        <h2>{risk_level}</h2>
                        <h3>Stroke Probability: {probability:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Display risk factors with icons - adjust style based on theme
                if theme == translations[language]["dark_mode"]:
                    st.markdown("""
                    <style>
                    .risk-factor {
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 5px;
                        background-color: #2E2E2E;
                        color: white;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='color: #ffffff;'>Risk Factors:</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <style>
                    .risk-factor {
                        padding: 10px;
                        margin: 5px 0;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                        color: #2c3e50;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='color: #2c3e50;'>Risk Factors:</h3>", unsafe_allow_html=True)
                
                if age > 60:
                    st.markdown("<div class='risk-factor'>Advanced Age</div>", unsafe_allow_html=True)
                if hypertension:
                    st.markdown("<div class='risk-factor'>Hypertension</div>", unsafe_allow_html=True)
                if heart_disease:
                    st.markdown("<div class='risk-factor'>Heart Disease</div>", unsafe_allow_html=True)
                if bmi > 30:
                    st.markdown("<div class='risk-factor'>High BMI</div>", unsafe_allow_html=True)
                if glucose > 140:
                    st.markdown("<div class='risk-factor'>High Glucose Level</div>", unsafe_allow_html=True)
                if smoking_status == translations[language]["smokes"]:
                    st.markdown("<div class='risk-factor'> Smoking</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    

elif page == translations[language]["about"]:
    
    st.title(translations[language]["about_title"])
    st.write(translations[language]["about_description"])

    st.subheader(translations[language]["project_overview"])
    if language == "English" :
       st.write(
        """
        Stroke is one of the leading causes of death and disability worldwide.
        Early detection of stroke risk factors can help in taking preventive measures.
        This project aims to build a user-friendly AI-powered tool
        that provides quick and reliable stroke risk predictions
        based on patient information such as age, medical history, and lifestyle habits.
        """)
    else:
        st.write( 
         """
        السكتة الدماغية هي واحدة من الأسباب الرئيسية للوفاة والإعاقة في جميع أنحاء العالم.
        يمكن أن يساعد الاكتشاف المبكر لعوامل خطر السكتة الدماغية في اتخاذ تدابير وقائية.
        يهدف هذا المشروع إلى بناء أداة تعتمد على الذكاء الاصطناعي
        توفر توقعات سريعة وموثوقة لمخاطر السكتة الدماغية
        بناءً على معلومات المريض مثل العمر والتاريخ الطبي وعادات الحياة.
        """)

    st.subheader(translations[language]["how_it_works"])
    if language == "English":
       st.write(
        """
        - The user enters patient details such as age, hypertension status, heart disease, BMI, glucose levels, etc-
        - The input data is processed and passed to a pre-trained Machine Learning model.
        - The model analyzes the data and returns a stroke risk prediction.
        - This prediction helps healthcare professionals or individuals assess potential risks and take preventive actions.
        """)
        
    else:
        st.write(
          """
         يقوم المستخدم بإدخال بيانات المريض مثل العمر، حالة ارتفاع ضغط الدم، أمراض القلب، مؤشر كتلة الجسم، مستويات الجلوكوز،إلخ.
         تتم معالجة البيانات المدخلة وتمريرها إلى نموذج تعلم آلي مدرّب مسبقًا.
         يقوم النموذج بتحليل البيانات وإرجاع توقع لمخاطر السكتة الدماغية.
         يساعد هذا التوقع الأطباء أو الأفراد على تقييم المخاطر المحتملة واتخاذ التدابير الوقائية اللازمة.
        """)

    st.subheader(translations[language]["why_it_matters"])
    if language == "English":
       st.write(
             
        """
        - *Early Detection* : Helps individuals and doctors take preventive measures.
        - *AI-Driven* : Uses Machine Learning for accurate and data-driven insights.
        - *Easy to Use* : A simple web interface for quick predictions.
        - *Scalable* : Can be improved with more data and better models in the future.
        """) 

    else :
        st.write(
              
        """
         الاكتشاف المبكر : يساعد الأفراد والأطباء في اتخاذ تدابير وقائية

         يعتمد على الذكاء الاصطناعي : يستخدم تعلم الآلة لتقديم تنبؤات دقيقة قائمة على البيانات

         سهل الاستخدام : واجهة ويب بسيطة توفر توقعات سريعة

         قابل للتطوير : يمكن تحسينه ببيانات أكثر ونماذج أفضل في المستقبل

        """)
