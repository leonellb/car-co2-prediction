import streamlit as st
import joblib
import json
import pandas as pd

from config import ELECTRIC_MODEL_FILE, COMBUSTION_MODEL_FILE, ELECTRIC_COLS_VALS_SELECTION_FILE, ELECTRIC_PREDICTION_PRESELECT_FILE, TRAIN_TEST_SPLIT_ELECTRIC_FILE, SCALER_ELECTRIC_FILE

# TODO replace/unify joblib functions

@st.cache_resource
def load_model(file_path):
    """Load a model from a .pkl file using joblib."""
    return joblib.load(file_path)

@st.cache_resource
def load_json(file_path):
    """Load variable selection from a .json file."""
    return json.load(open(file_path, 'r', encoding='utf-8'))

@st.cache_resource
def load_electric_test_train_split(file_path):
    """Load from a joblib file."""
    return joblib.load(file_path)

@st.cache_resource
def load_electric_scaler(file_path):
    """Load scaler from a joblib file."""
    return joblib.load(file_path)

# Set up the Streamlit app
st.set_page_config(page_title="Model Selector", layout="wide")

# Sidebar for model selection
st.sidebar.title("Model Selector")
sidebar_sel_pred_electric = "Electric Model Prediction"
sidebar_sel_pred_test_electric = "Electric Model Prediction of Test Data"
sidebar_sel_pred_combustion = "Combustion Model Prediction"
model_type = st.sidebar.radio(
    "Choose a model type:",
    (
        sidebar_sel_pred_electric,
        # sidebar_sel_pred_test_electric,
        sidebar_sel_pred_combustion,
    )
)

# Main page content
# st.title("Predicting Vehicle CO₂ Emissions")
   
if model_type == sidebar_sel_pred_electric:
    st.title("Predicting electrical energy consumption")
    st.write("Provide input values for live predictions:")

    # Load the Electric Model
    electric_model = load_model(ELECTRIC_MODEL_FILE)

    # Load categorical var selection
    electric_cols_val_selection = load_json(ELECTRIC_COLS_VALS_SELECTION_FILE)

    # Input fields for Electric Model
    # for col, vals in electric_cols_val_selection.items():
    #     # st.write(col)
    #     # selectbox if list
    #     if isinstance(vals, list):
    #         st.selectbox(col, electric_cols_val_selection[col])
    #     # elif isinstance(vals, float) or isinstance(vals, int):
    #     #     st.number_input(col, value=vals)
    #     else:
    #         st.number_input(col, value=vals)

    # member_state
    choice_member_state = st.selectbox("Member State", electric_cols_val_selection['member_state'])
    # manufacturer_name_eu
    choice_manufacturer_name_eu = st.selectbox("Manufacturer", electric_cols_val_selection['manufacturer_name_eu'])
    # vehicle_type
    choice_vehicle_type = st.selectbox("vehicle_type", electric_cols_val_selection['vehicle_type'])
    # commercial_name
    choice_commercial_name = st.selectbox("commercial_name", electric_cols_val_selection['commercial_name'])
    # category_of_vehicle
    choice_category_of_vehicle = st.selectbox("category_of_vehicle", electric_cols_val_selection['category_of_vehicle'])

    choice_mass_vehicle = st.number_input("mass_vehicle [kg]", value=electric_cols_val_selection['mass_vehicle'])
    choice_engine_power = st.number_input("engine_power [KW]", value=electric_cols_val_selection['engine_power'])
    choice_year = st.number_input("year", value=electric_cols_val_selection['year'])
    choice_electric_range = st.number_input("electric_range [km]", value=electric_cols_val_selection['electric_range'])

    # Prepare input for prediction with ELECTRIC_PREDICTION_PRESELECT_FILE

    preselection_data = load_json(ELECTRIC_PREDICTION_PRESELECT_FILE)
    # st.write(preselection_data)

    # initialize prediction dataframe
    prediction_dict = dict()
    for col, val in preselection_data.items():
        prediction_dict[col] = [val]
    df_prediction = pd.DataFrame.from_dict(prediction_dict)

    # Make prediction
    if st.button("Predict", type="primary"):
        # st.write("Calculating...")
        # st.write(choice_mass_vehicle)

        # update df_prediction based on user_input
        # choice_member_state
        df_prediction['member_state_'+choice_member_state] = 1
        # choice_manufacturer_name_eu
        df_prediction['manufacturer_name_eu_'+choice_manufacturer_name_eu] = 1
        # choice_vehicle_type
        df_prediction['vehicle_type_'+choice_vehicle_type] = 1
        # choice_commercial_name
        df_prediction['commercial_name_'+choice_commercial_name] = 1
        # choice_category_of_vehicle
        df_prediction['category_of_vehicle_'+choice_category_of_vehicle] = 1
        # choice_mass_vehicle
        df_prediction['mass_vehicle'] = choice_mass_vehicle
        # choice_engine_power
        df_prediction['engine_power'] = choice_engine_power
        # choice_year
        df_prediction['year'] = choice_year
        # choice_electric_range
        df_prediction['electric_range'] = choice_electric_range

        # scale the input data
        electric_scaler = load_electric_scaler(SCALER_ELECTRIC_FILE)

        cols_to_be_scaled = df_prediction[["mass_vehicle", "engine_power", "year", "electric_range"]].columns
        df_prediction[cols_to_be_scaled] = electric_scaler.transform(df_prediction[cols_to_be_scaled])

        # st.write(df_prediction[["mass_vehicle", "engine_power", "year", "electric_range"]])

        prediction = electric_model.predict(df_prediction)
        st.write("Prediction results electrical energy consumption [Wh/km]:")
        st.write(prediction)

if model_type == sidebar_sel_pred_test_electric:
    st.title("Predicting electrical energy consumption based on test data")
    
    # Load the Electric Model
    electric_model = load_model(ELECTRIC_MODEL_FILE)
    
    _, X_test, _, y_test = load_electric_test_train_split(TRAIN_TEST_SPLIT_ELECTRIC_FILE)

    # Pick the columns you want to show in the multiselect label
    display_cols = ['mass_vehicle', 'engine_power', 'year', 'electric_range']
    
    options = {}
    for i in X_test.head(100).index:
        values = [f"{col}={X_test.at[i, col]}" for col in display_cols]
        label = f"Index {i}: " + ", ".join(values)
        options[label] = i

    # Multiselect with custom labels
    selected_labels = st.multiselect(
        "Select rows based on features:",
        options=list(options.keys())
    )

    if selected_labels:
        selected_indices = [options[label] for label in selected_labels]
        selected_rows = X_test.loc[selected_indices]
        st.write("Selected Rows:")
        st.dataframe(selected_rows)
        # Make predictions
        predictions = electric_model.predict(selected_rows)
        results = selected_rows.copy()
        results["Prediction"] = predictions
        st.write("Prediction Results:")
        st.dataframe(results)

    # st.write(X_test.columns)
    # # Display multiselect for test rows
    # test_rows = st.multiselect(
    #     "Select test rows for prediction:",
    #     options=X_test.head(100).index.tolist(),
    #     # format_func=lambda idx: f"Row {idx}"
    # )

    # # Filter selected rows
    # if test_rows:
    #     selected_X_test = X_test.loc[test_rows]

    #     # Display selected rows
    #     st.write("Selected Test Data:")
    #     st.write(selected_X_test)

    #     # Make predictions
    #     predictions = electric_model.predict(selected_X_test)
    #     st.write("Prediction results electrical energy consumption [Wh/km]:")
    #     st.write(predictions)

elif model_type == sidebar_sel_pred_combustion:
    st.title("Predicting Carbon Dioxide Emissions")
    st.write("Provide input values for live predictions:")

    # Load the Combustion Model
    combustion_model = load_model(COMBUSTION_MODEL_FILE)

    # Input fields for Combustion Model
    fuel_rate = st.number_input("Fuel Rate (L/h)", value=5.0)
    efficiency = st.number_input("Efficiency (%)", value=30.0)

    # Prepare input for prediction
    preselection_data = [[fuel_rate, efficiency]]

    # Make prediction
    energy_output_prediction = combustion_model.predict(preselection_data)[0]
    st.write(f"Predicted Energy Output: {energy_output_prediction} kWh")