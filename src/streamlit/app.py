import joblib

import streamlit as st

# Set up the Streamlit app
st.set_page_config(page_title="Model Selector Predicting Vehicle CO₂ Emissions", layout="wide")

# Sidebar for model selection
st.sidebar.title("Model Selector")
model_type = st.sidebar.radio(
    "Choose a model type:",
    ("Electric Model", "Combustion Model")
)

# Main page content
st.title("Welcome to the Model Selector App")

if model_type == "Electric Model":
    st.subheader("You have selected the Electric Model")
    @st.cache_resource
    def load_model(model_name):
        """Load a model from a .pkl file using joblib."""
        return joblib.load(f"models/{model_name}.pkl")

    if model_type == "Electric Model":
        st.write("Provide input values for live predictions:")

        # Load the Electric Model
        electric_model = load_model("electric_model")

        # Input fields for Electric Model
        voltage = st.number_input("Voltage (V)", value=220.0)
        current = st.number_input("Current (A)", value=10.0)

        # Prepare input for prediction
        input_data = [[voltage, current]]

        # Make prediction
        power_prediction = electric_model.predict(input_data)[0]
        st.write(f"Predicted Power: {power_prediction} Watts")

    elif model_type == "Combustion Model":
        st.write("Provide input values for live predictions:")

        # Load the Combustion Model
        combustion_model = load_model("combustion_model")

        # Input fields for Combustion Model
        fuel_rate = st.number_input("Fuel Rate (L/h)", value=5.0)
        efficiency = st.number_input("Efficiency (%)", value=30.0)

        # Prepare input for prediction
        input_data = [[fuel_rate, efficiency]]

        # Make prediction
        energy_output_prediction = combustion_model.predict(input_data)[0]
        st.write(f"Predicted Energy Output: {energy_output_prediction} kWh")
