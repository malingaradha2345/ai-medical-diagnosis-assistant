import streamlit as st

st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

st.sidebar.title("🏥 AI Medical Diagnosis Assistant")

st.sidebar.markdown("---")

st.sidebar.write("### Diseases")
st.sidebar.write("❤️ Heart Disease")
st.sidebar.write("🩸 Diabetes")
st.sidebar.write("🫀 Liver Disease")
st.sidebar.write("🧠 Parkinson's Disease")
st.sidebar.write("🩺 Kidney Disease")

st.sidebar.markdown("---")

st.sidebar.write("### Technologies")
st.sidebar.write("""
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
""")

st.sidebar.markdown("---")

st.sidebar.info("Select a disease page from the sidebar menu.")

st.title("🏥 AI Medical Diagnosis Assistant")

st.write("""
Welcome to the **AI Medical Diagnosis Assistant**.

This application uses Machine Learning models to predict the possibility of different diseases based on the patient's health information.

Select a disease from the **sidebar** to begin prediction.
""")

st.markdown("---")

st.header("Diseases Supported")

col1, col2 = st.columns(2)

with col1:
    st.success("❤️ Heart Disease")
    st.success("🩸 Diabetes")
    st.success("🫀 Liver Disease")

with col2:
    st.success("🧠 Parkinson's Disease")
    st.success("🩺 Kidney Disease")

st.info("👈 Select a disease from the sidebar to start prediction.")