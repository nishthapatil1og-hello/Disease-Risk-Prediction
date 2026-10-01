
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Disease Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Risk Prediction")
st.write("Enter your details to view the model's predicted risk category.")

@st.cache_resource
def load_model():
    return joblib.load("decision_tree_model.pkl")

model = load_model()

with st.form("risk_form"):
    st.subheader("Personal Details")

    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Female", "Male"])

    st.subheader("Health Details")

    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=22.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=500, value=180)
    glucose = st.number_input("Glucose", min_value=40, max_value=400, value=90)
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=220, value=72)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Use", ["No", "Yes"])
    physical_activity = st.selectbox(
        "Physical Activity", ["Low", "Medium", "High"]
    )
    family_history = st.selectbox("Family History", ["No", "Yes"])
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_data = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Glucose": glucose,
        "Heart_Rate": heart_rate,
        "Sleep_Hours": sleep_hours,
        "Gender_Male": int(gender == "Male"),
        "Smoking_Yes": int(smoking == "Yes"),
        "Alcohol_Yes": int(alcohol == "Yes"),
        "Physical_Activity_Low": int(physical_activity == "Low"),
        "Physical_Activity_Medium": int(physical_activity == "Medium"),
        "Family_History_Yes": int(family_history == "Yes"),
        "Stress_Level_Low": int(stress_level == "Low"),
        "Stress_Level_Medium": int(stress_level == "Medium")
    }])

    input_data = input_data[model.feature_names_in_]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.info(f"Model-predicted category: **{prediction}**")

    st.caption(
        "This is an educational ML project, not a medical diagnosis. "
        "Predictions depend on the training data and model performance."
    )


