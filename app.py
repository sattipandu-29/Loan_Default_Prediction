import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/loan_model.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏦 Loan Prediction System")
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Major Project**

    Loan Default Prediction Using Machine Learning

    **Algorithm**
    - Random Forest Classifier

    **Language**
    - Python

    **Framework**
    - Streamlit
    """
)

# -----------------------------
# Main Title
# -----------------------------
st.title("🏦 Loan Default Prediction System")
st.write("Predict whether a loan application is likely to be approved based on applicant information.")

st.markdown("---")

# -----------------------------
# Input Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Amount Term", value=360.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

# -----------------------------
# Manual Encoding
# -----------------------------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

property_area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[property_area]

dependents = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}[dependents]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Loan Status"):

    input_data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ])

    prediction = model.predict(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

st.markdown("---")
st.caption("Developed using Python, Scikit-learn, Streamlit, and Random Forest.")

st.markdown("## 📊 Project Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", "614")

with col2:
    st.metric("Features", "11")

with col3:
    st.metric("Model Accuracy", "75.61%")


st.markdown("---")

st.subheader("📖 About Project")

st.write("""
This project predicts whether a loan application is likely to be approved
using Machine Learning.

The model is trained using historical loan application data.

### Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

### Machine Learning Algorithm
Random Forest Classifier
""")


st.info("""
💡 Tip:
A good credit history and stable income generally improve the chances
of loan approval.
""")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/loan_model.pkl")

# Model Accuracy
accuracy = 75.61
st.subheader("📈 Model Performance")

st.progress(accuracy / 100)

st.write(f"### Accuracy : {accuracy:.2f}%")