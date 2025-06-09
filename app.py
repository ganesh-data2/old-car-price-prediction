import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load('car_price_rfrmm_model.pkl')

st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("🚗 Car Price Prediction App")

st.markdown("Provide the car details below to estimate its selling price.")

# User input fields
company = st.selectbox("Select Company", ['Hyundai', 'Maruti', 'Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Nissan', 'Other'])
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, step=1)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=100)
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
ownership = st.selectbox("Ownership", ['First', 'Second', 'Third', 'Fourth & Above'])
engine = st.number_input("Engine Capacity (CC)", min_value=500, max_value=5000, step=100)

# Process inputs
if st.button("Predict Selling Price"):
    try:
        # Encode categorical variables manually or with prior mapping used in training
        data = pd.DataFrame({
            'company': [company],
            'year': [year],
            'fuel_type': [fuel_type],
            'kms_driven': [kms_driven],
            'transmission': [transmission],
            'ownership': [ownership],
            'engine': [engine]
        })

        # Optional: same encoding as used in training
        # You MUST match preprocessing done during model training

        # Example: Label encoding (adjust based on your model training)
        mappings = {
            'company': {'Maruti': 0, 'Hyundai': 1, 'Honda': 2, 'Toyota': 3, 'Ford': 4,
                        'BMW': 5, 'Audi': 6, 'Nissan': 7, 'Other': 8},
            'fuel_type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4},
            'transmission': {'Manual': 0, 'Automatic': 1},
            'ownership': {'First': 0, 'Second': 1, 'Third': 2, 'Fourth & Above': 3}
        }

        for col in ['company', 'fuel_type', 'transmission', 'ownership']:
            data[col] = data[col].map(mappings[col])

        # Predict the price
        prediction = model.predict(data)[0]

        st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
