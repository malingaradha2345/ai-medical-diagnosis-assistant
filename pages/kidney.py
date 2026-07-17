import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Kidney Disease Prediction")
st.write("Enter the patient's details below.")

# Load model and scaler
model = joblib.load("models/kidney_model.pkl")
scaler = joblib.load("preprocessing/kidney_scaler.pkl")

# Mapping dictionaries
yes_no = {"No": 0, "Yes": 1}
normal_abnormal = {"Normal": 0, "Abnormal": 1}
present_notpresent = {"Not Present": 0, "Present": 1}
good_poor = {"Good": 0, "Poor": 1}

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.005, 1.025, 1.020, format="%.3f")
    al = st.number_input("Albumin", 0, 5, 0)
    su = st.number_input("Sugar", 0, 5, 0)

    rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
    pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
    ba = st.selectbox("Bacteria", ["Not Present", "Present"])

with col2:
    bgr = st.number_input("Blood Glucose Random", 0, 500, 120)
    bu = st.number_input("Blood Urea", 0.0, 400.0, 40.0)
    sc = st.number_input("Serum Creatinine", 0.0, 20.0, 1.2)
    sod = st.number_input("Sodium", 50.0, 200.0, 138.0)
    pot = st.number_input("Potassium", 1.0, 10.0, 4.5)
    hemo = st.number_input("Hemoglobin", 0.0, 20.0, 13.5)
    pcv = st.number_input("Packed Cell Volume", 0, 60, 44)
    wc = st.number_input("White Blood Cell Count", 0, 30000, 8000)
    rc = st.number_input("Red Blood Cell Count", 0.0, 10.0, 5.0)

    htn = st.selectbox("Hypertension", ["No", "Yes"])
    dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
    cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
    appet = st.selectbox("Appetite", ["Good", "Poor"])
    pe = st.selectbox("Pedal Edema", ["No", "Yes"])
    ane = st.selectbox("Anemia", ["No", "Yes"])

if st.button("Predict"):

    input_data = np.array([[
        age,
        bp,
        sg,
        al,
        su,
        normal_abnormal[rbc],
        normal_abnormal[pc],
        present_notpresent[pcc],
        present_notpresent[ba],
        bgr,
        bu,
        sc,
        sod,
        pot,
        hemo,
        pcv,
        wc,
        rc,
        yes_no[htn],
        yes_no[dm],
        yes_no[cad],
        good_poor[appet],
        yes_no[pe],
        yes_no[ane]
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Kidney Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Recommendations")
        st.write("• Consult a nephrologist.")
        st.write("• Drink adequate water as advised by your doctor.")
        st.write("• Reduce salt intake.")
        st.write("• Control blood pressure and blood sugar.")
        st.write("• Avoid smoking and alcohol.")
        st.write("• Follow prescribed medications and regular kidney checkups.")

    else:
        st.success("✅ No Kidney Disease Detected")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Health Tips")
        st.write("• Stay hydrated.")
        st.write("• Maintain a healthy diet.")
        st.write("• Exercise regularly.")
        st.write("• Avoid excessive painkiller use.")
        st.write("• Monitor blood pressure and blood sugar regularly.")