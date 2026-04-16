import streamlit as st
import numpy as np
import joblib
import pandas as pd

def model():
    return joblib.load('Crop_recommendation.pkl')
model = joblib.load('Crop_recommendation.pkl')
st.title('🌾Crop Recommendation Project using ML🌾')

st.write("Enter soil and environmental conditions:")

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N)", 0.0)
    p = st.number_input("Phosphorus (P)", 0.0)
    k = st.number_input("Potassium (K)", 0.0)
    temp = st.number_input("Temperature (°C)", 0.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0)
    ph = st.number_input("pH", 0.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0)

# Optional state input
state = st.text_input("State (optional)")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🌱 Predict Crop"):
   # Base features
    input_dict = {
        'N': n,
        'P': p,
        'K': k,
        'temperature': temp,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)

    st.success(f"✅ Recommended Crop: {prediction[0]}")

    # Confidence score
    try:
        prob = model.predict_proba(df)
        confidence = np.max(prob)
        st.info(f"Confidence: {confidence:.2f}")
    except:
        pass

        st.markdown("---")

if st.checkbox("Show Feature Importance"):
    try:
        importance = model.feature_importances_
        features = df.columns

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(imp_df.set_index("Feature"))
    except:
        st.warning("Feature importance not available.")
st.markdown("---")
st.caption("Built with Streamlit 🚀")
