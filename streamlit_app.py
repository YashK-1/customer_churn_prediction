import streamlit as st
import pandas as pd
import pickle
import os

# Load model
MODEL_PATH = os.path.join("model", "final_pipeline.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# App Title
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction App")

st.markdown("Enter customer details in the sidebar to predict churn.")

# Sidebar form
st.sidebar.header("📋 Customer Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2500.0)

# Submit button
if st.sidebar.button("Predict Churn"):
    input_data = pd.DataFrame([{
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges
    }])

    prediction = model.predict(input_data)[0]
    result = "Yes 🔴 - Likely to Churn" if prediction == 1 else "No 🟢 - Not Likely to Churn"
    st.subheader("Prediction Result")
    st.success(result)
