import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/rf_model.joblib")
median_values = joblib.load("models/median_values.joblib")

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Classifier")
st.markdown("Predict whether a wine sample is **Good** (quality ≥ 7) or **Not Good** based on its chemical properties.")

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Center the form using columns
col1, col2, col3 = st.columns([1, 2, 1])  # left, middle, right

with col2:  # form in the middle
    st.header("🔬 Enter Wine Attributes")
    inputs = []
    with st.form("prediction_form"):
        for i, feature in enumerate(feature_names):
            value = st.number_input(f"{feature}", value=float(median_values[i]))
            inputs.append(value)

        submit = st.form_submit_button("🔎 Predict")

# Prepare input
X_input = np.array(inputs).reshape(1, -1)

# Show prediction result
if submit:
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    confidence = proba[1] if prediction == 1 else proba[0]

    col1, col2, col3 = st.columns([1, 2, 1])  # new row for result
    with col2:
        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color:#d4edda;padding:20px;border-radius:10px;text-align:center;">
                    <h3 style="color:#155724;">✅ This wine is <b>Good Quality</b></h3>
                    <p>Confidence: <b>{confidence:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)  # progress bar under result
        else:
            st.markdown(
                f"""
                <div style="background-color:#f8d7da;padding:20px;border-radius:10px;text-align:center;">
                    <h3 style="color:#721c24;">❌ This wine is <b>Not Good Quality</b></h3>
                    <p>Confidence: <b>{confidence:.2f}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)  # progress bar under result
