import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("model/best_heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            color: #333333;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #ff2e2e;
            color: white;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            border-radius: 8px;
        }
        h1 {
            color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Title and Intro
# -----------------------
st.title("❤️ AI Health Risk Predictor")
st.markdown("""
Predict your **heart disease risk** in seconds!
Fill in your health details below and get **instant AI predictions** along with risk probability.
""")
st.divider()

# -----------------------
# Sidebar Information
# -----------------------
st.sidebar.header("ℹ️ About This App")
st.sidebar.info("""
This AI-powered health predictor uses **Machine Learning** to estimate your risk of heart disease.
**Features used:** Age, Sex, Chest Pain, Blood Pressure, Cholesterol, Heart Rate, Exercise-induced Angina, etc.
**Purpose:** Personal awareness — **not a medical diagnosis**. Always consult a doctor.
""")

# -----------------------
# Input Section with Columns
# -----------------------
st.markdown("### 👇 Enter Your Health Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])

with col2:
    restecg = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", [0, 1, 2])

# Convert categorical inputs to numeric
sex_value = 1 if sex == "Male" else 0
fbs_value = 1 if fbs == "Yes" else 0
exang_value = 1 if exang == "Yes" else 0

user_input = np.array([[age, sex_value, cp, trestbps, chol, fbs_value, restecg,
                        thalach, exang_value, oldpeak, slope, ca, thal]])
scaled_input = scaler.transform(user_input)

# -----------------------
# Prediction Button
# -----------------------
if st.button("🔍 Predict My Heart Risk"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100  # Probability of risk

    st.divider()
    st.markdown("### 🩺 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! Risk Probability: **{probability:.2f}%**")
        st.markdown("Please consult a **cardiologist** and maintain a healthy lifestyle. ❤️")
    else:
        st.success(f"✅ Low Risk! Risk Probability: **{probability:.2f}%**")
        st.markdown("Keep up your **healthy lifestyle** and regular checkups! 🏃‍♂️🥗")

    # -----------------------
    # Probability Visualization
    # -----------------------
    st.markdown("### 📊 Risk Probability Visualization")
    fig, ax = plt.subplots()
    ax.bar(["Low Risk", "High Risk"], [100 - probability, probability], color=["#4CAF50", "#FF4B4B"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    for i, v in enumerate([100 - probability, probability]):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig)

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px'>
Developed by **Harshvardhan Das** | Powered by **Machine Learning & Streamlit**
</div>
""", unsafe_allow_html=True)
