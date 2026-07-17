import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩸", layout="wide")

st.title("🩸 Diabetes Prediction")

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("preprocessing/diabetes_scaler.pkl")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 33)

if st.button("Predict"):
    x = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    x = scaler.transform(x)
    pred = model.predict(x)[0]
    conf = model.predict_proba(x)[0][pred] * 100

    if pred == 1:
        st.error("🩸 Diabetes Detected")
        st.info(f"Confidence: {conf:.2f}%")
        st.subheader("Recommendations")
        st.write("• Consult a physician.")
        st.write("• Monitor blood sugar regularly.")
        st.write("• Follow a balanced diet.")
        st.write("• Exercise regularly.")
        st.write("• Take medicines as prescribed.")
    else:
        st.success("💚 No Diabetes Detected")
        st.info(f"Confidence: {conf:.2f}%")
        st.subheader("Health Tips")
        st.write("• Maintain a healthy weight.")
        st.write("• Stay physically active.")
        st.write("• Eat nutritious food.")
        st.write("• Have regular health checkups.")

