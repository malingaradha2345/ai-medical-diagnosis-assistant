import streamlit as st
import joblib
import numpy as np
from utils.shap_helper import show_shap

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction")

model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("preprocessing/scaler.pkl")

sex_map={"Female":0,"Male":1}
cp_map={"Asymptomatic":0,"Atypical angina":1,"Non-anginal pain":2,"Typical angina":3}
fbs_map={"Greater than 120 mg/ml":0,"Lower than 120 mg/ml":1}
ecg_map={"Left ventricular hypertrophy":0,"Normal":1,"ST-T wave abnormality":2}
angina_map={"No":0,"Yes":1}
slope_map={"Downsloping":0,"Flat":1,"Upsloping":2}
vessel_map={"Four":0,"One":1,"Three":2,"Two":3,"Zero":4}
thal_map={"Fixed Defect":0,"No":1,"Normal":2,"Reversable Defect":3}

c1,c2=st.columns(2)
with c1:
    age=st.number_input("Age",1,120,40)
    sex=st.selectbox("Sex",["Male","Female"])
    cp=st.selectbox("Chest Pain Type",["Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"])
    rbp=st.number_input("Resting Blood Pressure",50,250,120)
    chol=st.number_input("Cholesterol",50,700,200)
    fbs=st.selectbox("Fasting Blood Sugar",["Greater than 120 mg/ml","Lower than 120 mg/ml"])
    recg=st.selectbox("Rest ECG",["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
with c2:
    hr=st.number_input("Maximum Heart Rate",50,250,150)
    ex=st.selectbox("Exercise Induced Angina",["Yes","No"])
    old=st.number_input("Old Peak",0.0,10.0,1.0)
    slope=st.selectbox("ST Slope",["Upsloping","Flat","Downsloping"])
    vessels=st.selectbox("Number of Major Vessels",["Zero","One","Two","Three","Four"])
    thal=st.selectbox("Thalassemia",["Normal","Fixed Defect","Reversable Defect","No"])

if st.button("Predict"):
    x=np.array([[
        age,sex_map[sex],cp_map[cp],rbp,chol,fbs_map[fbs],ecg_map[recg],
        hr,angina_map[ex],old,slope_map[slope],vessel_map[vessels],thal_map[thal]
    ]])
    xs=scaler.transform(x)
    pred=model.predict(xs)[0]
    conf=model.predict_proba(xs)[0][pred]*100
    if pred==1:
        st.error("❤️ Heart Disease Detected")
        st.info(f"Confidence: {conf:.2f}%")
        st.subheader("Recommendations")
        st.write("• Consult a cardiologist")
        st.write("• Eat a healthy diet")
        st.write("• Exercise as advised by your doctor")
        st.write("• Monitor blood pressure and cholesterol")
        st.write("• Avoid smoking and excessive alcohol")
    else:
        st.success("💚 No Heart Disease Detected")
        st.info(f"Confidence: {conf:.2f}%")
        st.subheader("Health Tips")
        st.write("• Maintain a healthy lifestyle")
        st.write("• Exercise regularly")
        st.write("• Eat a balanced diet")
        st.write("• Get regular health checkups")

