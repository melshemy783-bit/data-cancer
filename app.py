import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction")
st.write("Enter the patient's features and click Predict.")

feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

user_input = {}

for feature in feature_names:
    user_input[feature] = st.number_input(feature, value=0.0)


if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.error("🔴 Malignant (Cancer)")
    else:
        st.success("🟢 Benign (Not Cancer)")

    st.subheader("Prediction Probability")

    st.write(f"Malignant : {probability[0]*100:.2f}%")
    st.write(f"Benign : {probability[1]*100:.2f}%")