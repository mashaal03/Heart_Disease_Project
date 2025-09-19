import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD THE SAVED MODEL AND SCALER ---
# Make sure the paths are correct relative to where you run `streamlit run`
try:
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'final_model.pkl' and 'scaler.pkl' are in the 'models' directory.")
    st.stop()


# --- 2. DEFINE THE FEATURE ORDER AND NUMERICAL/CATEGORICAL FEATURES ---
# This MUST be the same order as the columns used for training the model
# From Step 3 (03_feature_selection.ipynb), this was our final list.
# IMPORTANT: Update this list if you used different features!
FINAL_FEATURES_ORDER = [
    'oldpeak', 'thalach', 'age', 'chol', 'trestbps',
    'cp_3.0', 'exang_1.0', 'ca_1.0', 'thal_7.0', 'sex_1.0'
]

# Identify which of these are numerical for scaling
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# --- 3. CREATE THE STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title('🩺 Heart Disease Prediction App')
st.write("This app predicts the likelihood of a patient having heart disease based on their medical data. Please enter the patient's information in the sidebar.")

st.sidebar.header('Patient Input Features')

def user_input_features():
    """Creates sidebar inputs and returns a dictionary of features."""
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (chol)', 126, 564, 240)
    thalach = st.sidebar.slider('Max Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST depression induced by exercise (oldpeak)', 0.0, 6.2, 1.0)
    ca = st.sidebar.selectbox('Number of major vessels colored by flouroscopy (ca)', ('0', '1', '2', '3'))
    thal = st.sidebar.selectbox('Thallium Stress Test Result (thal)', ('Normal', 'Fixed defect', 'Reversible defect'))

    # --- Process inputs to match model's expected format ---
    # This section is critical for converting user-friendly inputs to the one-hot encoded format
    data = {
        'age': age,
        'trestbps': trestbps,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak,
        'sex_1.0': 1 if sex == 'Male' else 0,
        'cp_3.0': 1 if cp == 'Non-anginal Pain' else 0, # Example, adjust based on your features!
        'exang_1.0': 1 if exang == 'Yes' else 0,
        'ca_1.0': 1 if ca == '1' else 0, # Example, adjust based on your features!
        'thal_7.0': 1 if thal == 'Reversible defect' else 0 # Example, adjust based on your features!
    }
    return data

input_data = user_input_features()

st.subheader('Patient Input Summary')
st.write(pd.DataFrame([input_data]))

# --- 4. PREDICTION LOGIC ---
if st.button('Predict'):
    # Create a DataFrame from the input data in the correct feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FINAL_FEATURES_ORDER]

    # Scale the numerical features
    input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.error(f'High Risk of Heart Disease. (Probability: {prediction_proba[0][1]*100:.2f}%)')
        st.write("Based on the input data, the model predicts a high probability of heart disease. Please consult a medical professional for advice.")
    else:
        st.success(f'Low Risk of Heart Disease. (Probability: {prediction_proba[0][0]*100:.2f}%)')
        st.write("Based on the input data, the model predicts a low probability of heart disease. Continue to maintain a healthy lifestyle.")