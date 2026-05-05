import streamlit as st
import pickle
import numpy as np

# Load your trained model
model = pickle.load(open("Crop_recommendation (4).pkl", "rb"))

st.title("🌱 Crop Recommendation System")

st.write("Enter soil and environmental conditions to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH value")
rainfall = st.number_input("Rainfall (mm)")

# Prediction button
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)

    st.success(f"🌾 Recommended Crop: {prediction[0]}")
