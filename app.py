import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, imputer, and scaler
model = pickle.load(open("fraud_model.pkl", "rb"))
imputer = pickle.load(open("imputer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to check for fraud.")

# Input form
with st.form("fraud_form"):
    time = st.number_input("Transaction Time", value=0.0)
    amount = st.number_input("Transaction Amount", value=0.0)
    v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

    submitted = st.form_submit_button("Check Transaction")

# Predict after button click
if submitted:
    input_data = pd.DataFrame([[time] + v_features + [amount]],
                              columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    
    # Preprocessing
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Result
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")