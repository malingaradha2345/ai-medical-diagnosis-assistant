import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Liver Disease Prediction")
st.write("Enter the patient's details below.")

# Load model and scaler
model = joblib.load("models/liver_model.pkl")
scaler = joblib.load("preprocessing/liver_scaler.pkl")

gender_map = {
    "Male": 1,
    "Female": 0
}

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", 0.0, 80.0, 1.0)
    direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 30.0, 0.3)
    alkaline_phosphotase = st.number_input(
        "Alkaline Phosphotase",
        0,
        2500,
        200
    )

with col2:
    alamine = st.number_input(
        "Alamine Aminotransferase",
        0,
        2500,
        35
    )

    aspartate = st.number_input(
        "Aspartate Aminotransferase",
        0,
        2500,
        40
    )

    total_protiens = st.number_input(
        "Total Protiens",
        0.0,
        15.0,
        6.5
    )

    albumin = st.number_input(
        "Albumin",
        0.0,
        10.0,
        3.5
    )

    agr = st.number_input(
        "Albumin and Globulin Ratio",
        0.0,
        5.0,
        1.0
    )

if st.button("Predict"):

    input_data = np.array([[
        age,
        gender_map[gender],
        total_bilirubin,
        direct_bilirubin,
        alkaline_phosphotase,
        alamine,
        aspartate,
        total_protiens,
        albumin,
        agr
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Liver Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Recommendations")
        st.write("• Consult a hepatologist.")
        st.write("• Avoid alcohol completely.")
        st.write("• Follow a balanced diet.")
        st.write("• Drink plenty of water.")
        st.write("• Take medicines only as prescribed.")
        st.write("• Schedule regular liver function tests.")

    else:
        st.success("✅ No Liver Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Health Tips")
        st.write("• Eat healthy food.")
        st.write("• Exercise regularly.")
        st.write("• Limit fatty foods.")
        st.write("• Stay hydrated.")
        st.write("• Avoid smoking and alcohol.")