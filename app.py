# =====================================
# Breast Cancer Prediction Streamlit UI
# =====================================

import streamlit as st
import numpy as np
import joblib

# ==============================
# Load Model and Scaler
# ==============================

model = joblib.load("best_model.sav")

scaler = joblib.load("scaler.sav")

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# Title
# ==============================

st.title("🩺 Breast Cancer Prediction System")

st.write(
    "This Machine Learning model predicts whether the tumor is Malignant or Benign."
)

st.markdown("---")

# ==============================
# Sidebar
# ==============================

st.sidebar.title("About Project")

st.sidebar.info(
    """
    Breast Cancer Prediction using:
    
    - Machine Learning
    - Streamlit
    - Scikit-learn
    
    Models Used:
    - Logistic Regression
    - SVM
    - Random Forest
    """
)

# ==============================
# Input Section
# ==============================

st.subheader("Enter Tumor Details")

col1, col2 = st.columns(2)

# ==============================
# Column 1
# ==============================

with col1:

    radius_mean = st.number_input("Radius Mean")

    texture_mean = st.number_input("Texture Mean")

    perimeter_mean = st.number_input("Perimeter Mean")

    area_mean = st.number_input("Area Mean")

    smoothness_mean = st.number_input("Smoothness Mean")

    compactness_mean = st.number_input("Compactness Mean")

    concavity_mean = st.number_input("Concavity Mean")

    concave_points_mean = st.number_input("Concave Points Mean")

    symmetry_mean = st.number_input("Symmetry Mean")

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

    radius_se = st.number_input("Radius SE")

    texture_se = st.number_input("Texture SE")

    perimeter_se = st.number_input("Perimeter SE")

    area_se = st.number_input("Area SE")

    smoothness_se = st.number_input("Smoothness SE")

# ==============================
# Column 2
# ==============================

with col2:

    compactness_se = st.number_input("Compactness SE")

    concavity_se = st.number_input("Concavity SE")

    concave_points_se = st.number_input("Concave Points SE")

    symmetry_se = st.number_input("Symmetry SE")

    fractal_dimension_se = st.number_input("Fractal Dimension SE")

    radius_worst = st.number_input("Radius Worst")

    texture_worst = st.number_input("Texture Worst")

    perimeter_worst = st.number_input("Perimeter Worst")

    area_worst = st.number_input("Area Worst")

    smoothness_worst = st.number_input("Smoothness Worst")

    compactness_worst = st.number_input("Compactness Worst")

    concavity_worst = st.number_input("Concavity Worst")

    concave_points_worst = st.number_input("Concave Points Worst")

    symmetry_worst = st.number_input("Symmetry Worst")

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

# ==============================
# Predict Button
# ==============================

if st.button("Predict"):

    # Create Input Array
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,

        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,

        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Result
    st.markdown("---")

    if prediction[0] == 1:

        st.error("⚠️ The Breast Cancer is Malignant")

    else:

        st.success("✅ The Breast Cancer is Benign")

# ==============================
# Footer
# ==============================

st.markdown("---")

st.caption("Developed using Streamlit and Machine Learning")