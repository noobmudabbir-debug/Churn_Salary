import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load Preprocessor + Model
preprocessor = joblib.load("preprocessor.pkl")
model = tf.keras.models.load_model("salary_model.h5")

# -------------------------------
# 🎨 Streamlit App
# -------------------------------
st.set_page_config(page_title="💰 Salary Prediction App", layout="centered")

st.title("💰 Customer Salary Prediction (Pakistan Bank Range)")
st.write("Fill in the details below to estimate salary (20,000 – 150,000 PKR).")

# -------------------------------
# Input fields (Vertical)
# -------------------------------
geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("👤 Gender", ["Male", "Female"])
credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("📅 Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0, step=1000.0)
num_products = st.number_input("📦 Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active = st.selectbox("✅ Is Active Member", [0, 1])

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔮 Predict Salary"):
    # Create dataframe for input
    input_data = pd.DataFrame([{
        "Geography": geography,
        "Gender": gender,
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active
    }])

    # Transform using preprocessor
    input_processed = preprocessor.transform(input_data)

    # Predict
    pred_salary = model.predict(input_processed)[0][0]

    st.success(f"💰 Estimated Salary: **PKR {pred_salary:,.0f}**")
