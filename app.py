import streamlit as st
import pandas as pd
import joblib

# Load model safely
try:
    model = joblib.load("loan_prediction_model (1).pkl")
    encoder = joblib.load("label_encoder (1).pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to check loan approval status")

# User Inputs
gender = st.selectbox("Gender", encoder["Gender"].classes_)
married = st.selectbox("Married", encoder["Married"].classes_)
dependents = st.selectbox("Dependents", encoder["Dependents"].classes_)
education = st.selectbox("Education", encoder["Education"].classes_)
self_employed = st.selectbox("Self_Employed", encoder["Self_Employed"].classes_)

app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", encoder["Property_Area"].classes_)

df = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [app_income],
    "CoapplicantIncome": [coapp_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

if st.button("Predict Loan Status"):
    
    # Encode only categorical columns
    for col in encoder:
        if col in df.columns and col != "Credit_History":
            df[col] = encoder[col].transform(df[col])

    try:
        model_features = model.feature_names_in_

        for feature in model_features:
            if feature not in df.columns:
                df[feature] = 0

        df_final = df[model_features]

        prediction = model.predict(df_final)

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Current columns:", list(df.columns))
