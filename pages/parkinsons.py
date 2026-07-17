import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Parkinson's Disease Prediction")
st.write("Enter the patient's voice measurements below.")

# Load model and scaler
model = joblib.load("models/parkinsons_model.pkl")
scaler = joblib.load("preprocessing/parkinsons_scaler.pkl")

col1, col2 = st.columns(2)

with col1:
    mdvp_fo = st.number_input("MDVP:Fo(Hz)", value=120.0)
    mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", value=150.0)
    mdvp_flo = st.number_input("MDVP:Flo(Hz)", value=100.0)
    mdvp_jitter = st.number_input("MDVP:Jitter(%)", value=0.005)
    mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00004)
    mdvp_rap = st.number_input("MDVP:RAP", value=0.003)
    mdvp_ppq = st.number_input("MDVP:PPQ", value=0.003)
    jitter_ddp = st.number_input("Jitter:DDP", value=0.009)

with col2:
    mdvp_shimmer = st.number_input("MDVP:Shimmer", value=0.03)
    mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.28)
    shimmer_apq3 = st.number_input("Shimmer:APQ3", value=0.015)
    shimmer_apq5 = st.number_input("Shimmer:APQ5", value=0.020)
    mdvp_apq = st.number_input("MDVP:APQ", value=0.025)
    shimmer_dda = st.number_input("Shimmer:DDA", value=0.045)
    nhr = st.number_input("NHR", value=0.02)
    hnr = st.number_input("HNR", value=21.0)

# Remaining features
col3, col4 = st.columns(2)

with col3:
    rpde = st.number_input("RPDE", value=0.50)
    dfa = st.number_input("DFA", value=0.70)
    spread1 = st.number_input("Spread1", value=-5.0)
    spread2 = st.number_input("Spread2", value=0.25)

with col4:
    d2 = st.number_input("D2", value=2.30)
    ppe = st.number_input("PPE", value=0.20)

if st.button("Predict"):

    input_data = np.array([[
        mdvp_fo,
        mdvp_fhi,
        mdvp_flo,
        mdvp_jitter,
        mdvp_jitter_abs,
        mdvp_rap,
        mdvp_ppq,
        jitter_ddp,
        mdvp_shimmer,
        mdvp_shimmer_db,
        shimmer_apq3,
        shimmer_apq5,
        mdvp_apq,
        shimmer_dda,
        nhr,
        hnr,
        rpde,
        dfa,
        spread1,
        spread2,
        d2,
        ppe
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Parkinson's Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Recommendations")
        st.write("• Consult a neurologist.")
        st.write("• Take prescribed medications regularly.")
        st.write("• Perform regular physical exercise.")
        st.write("• Attend speech and occupational therapy if recommended.")
        st.write("• Maintain a healthy and balanced diet.")
        st.write("• Schedule regular medical follow-ups.")

    else:
        st.success("✅ No Parkinson's Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Health Tips")
        st.write("• Exercise regularly.")
        st.write("• Eat a balanced diet.")
        st.write("• Get adequate sleep.")
        st.write("• Reduce stress.")
        st.write("• Attend regular health check-ups.")