import json
import pickle
import joblib
import pandas as pd
from config import ELECTRIC_COLS_VALS_SELECTION_FILE, ELECTRIC_MODEL_FILE, ELECTRIC_PREDICTION_PRESELECT_FILE, SCALER_ELECTRIC_FILE, TRAIN_TEST_SPLIT_COMBUSTION_FILE, TRAIN_TEST_SPLIT_ELECTRIC_FILE, COMBUSTION_MODEL_FILE, RAW_DATA_COMBUSTION_FILE

import streamlit as st

# TODO replace/unify joblib functions

@st.cache_resource
def load_model(file_path):
    """Load a model from a .pkl file using joblib."""
    return joblib.load(file_path)

@st.cache_resource
def load_json(file_path):
    """Load variable selection from a .json file."""
    return json.load(open(file_path, encoding='utf-8'))

@st.cache_resource
def load_test_train_split(file_path):
    """Load from a joblib file."""
    return joblib.load(file_path)

@st.cache_resource
def load_scaler(file_path):
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
        electric_scaler = load_scaler(SCALER_ELECTRIC_FILE)

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

    _, X_test, _, y_test = load_test_train_split(TRAIN_TEST_SPLIT_ELECTRIC_FILE)

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
    import numpy as np
    import pickle
    from joblib import dump, load
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    #from xgboost import XGBRegressor

    # # Load the Combustion Model
    #combustion_model = load_model(COMBUSTION_MODEL_FILE)
    _, X_test_comb, _, y_test_comb = load_test_train_split(TRAIN_TEST_SPLIT_COMBUSTION_FILE)
    combustion_model = load('files/output/models/combustion_model.joblib')
    with open('files/raw-dataset_combustion.pkl', 'rb') as f:
        df_pe_cleaned = pickle.load(f)
    batch_si= int(0.1*df_pe_cleaned.shape[0])
    df_pe_cleaned= df_pe_cleaned[:batch_si]
    batch_size= int(0.1*X_test_comb.shape[0])
    X_test_batch= X_test_comb[:batch_size]
    y_test_batch=y_test_comb[:batch_size]
    # preparation for selctboxes
    # unique values of columns
    uniques_m_s = np.sort(df_pe_cleaned['member_state'].unique()[:20])
    choice_m_s= st.selectbox('Member_state:', uniques_m_s)
    uniques_m_n_e = np.sort(df_pe_cleaned['manufacturer_name_eu'].unique()[:20])
    choice_m_n_e= st.selectbox('Manufactorer_EU:', uniques_m_n_e)
    uniques_v_t = np.sort(df_pe_cleaned['vehicle_type'].unique()[:20])
    choice_v_t= st.selectbox('vehicle_type:', uniques_v_t)
    uniques_c_o_v = np.sort(df_pe_cleaned['category_of_vehicle'].unique()[:20])
    choice_c_o_v= st.selectbox('category of vehicle:', uniques_c_o_v)
    uniques_f_t = np.sort(df_pe_cleaned['fuel_type'].unique()[:20])
    choice_f_t= st.selectbox('fuel_type:', uniques_f_t)
    uniques_f_m = np.sort(df_pe_cleaned['fuel_mode'].unique()[:20])
    choice_f_m= st.selectbox('fuel_mode:', uniques_f_m)
    uniques_weltp_test_mass = np.sort(df_pe_cleaned['weltp_test_mass'].unique()[:20])
    formatted_values_test_mass = ["{:.1f}".format(x) for x in uniques_weltp_test_mass]
    choice_weltp_test_mass= st.selectbox('wltp_test_mass [kg]):',formatted_values_test_mass)
    # Field for free entry
    custom_value = st.text_input('Free input weltp_test_mass [kg]', key = 'weltp_test_mass')
    if custom_value:
        choice_weltp_test_mass = float(custom_value) 
    uniques_e_c = np.sort(df_pe_cleaned['engine_capacity'].unique()[:20])
    formatted_values_e_c = ["{:.1f}".format(x) for x in uniques_e_c]
    choice_e_c= st.selectbox("engine_capacity [cm^3] :", formatted_values_e_c)
    custom_value = st.text_input('Free input engine capacity [cm^3]', key= 'engine_capacity')
    if custom_value:
        choice_e_c = float(custom_value) 
    uniques_e_p = np.sort(df_pe_cleaned['engine_power'].unique()[:20])
    formatted_values_e_p = ["{:.1f}".format(x) for x in uniques_e_p]
    choice_e_p= st.selectbox('engine_power [kW]', formatted_values_e_p)
    custom_value = st.text_input('Free input engine power [kW]', key = 'engine_power')
    if custom_value:
        choice_e_p = float(custom_value)
    uniques_year = np.sort(df_pe_cleaned['year'].unique()[:20])
    choice_year= st.selectbox('year:', uniques_year)
    custom_value = st.text_input('Free input year', key = 'year')
    if custom_value:
        choice_year = custom_value
    #writing selected values to a dataframe
    
    if st.button("Predict", type="primary"):
        # st.write("Calculating...")
        # st.write(choice_mass_vehicle)

        # update df_prediction based on user_input
        # choice_member_state
        df_prediction_comb = pd.DataFrame({
        'member_state': [choice_m_s],
        'manufacturer_name_eu': [choice_m_n_e],
        'vehicle_type': [choice_v_t],
        'category_of_vehicle': [choice_c_o_v],
        'fuel_type': [choice_f_t],
        'fuel_mode': [choice_f_m],
        'weltp_test_mass': [choice_weltp_test_mass],
        'engine_capacity': [choice_e_c],
        'engine_power': [choice_e_p],
        'year': [choice_year]
        })
        X_pe_cleaned=df_pe_cleaned.drop(columns = ["specific_co2_emissions","fuel_consumption"])
        X_prediction_comb = pd.concat([X_pe_cleaned, df_prediction_comb], axis=0, ignore_index=True)
        #X_prediction_comb = X_prediction_comb.drop(columns = ["ID"])
        #labelencoding the inserted values
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        enc = LabelEncoder()
        scaler= StandardScaler()
        cat_cols = X_prediction_comb.select_dtypes(include="object").columns
        num_cols = X_prediction_comb.select_dtypes(include=["float64", "int64"]).columns
        st.write("cat_cols: ", cat_cols)
        st.write("num_cols: ", num_cols)
        for col in cat_cols:
            X_prediction_comb[col] = enc.fit_transform(X_prediction_comb[col])
        X_prediction_comb[num_cols]= scaler.fit_transform(X_prediction_comb[num_cols])
        y_pred = combustion_model.predict(X_prediction_comb.iloc[[-1]])
         
        # Display the selected rows 
        dict={"weltp_specific_co2_emissions": y_pred}
        results = pd.DataFrame(dict)
        st.write("Prediction results CO2 emissions [g CO2/100km]: ")
        st.write(results)
    if st.checkbox("Combustion model testing and metrics"):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        st.write("#### Cobustion model XGBRegressor and metrics")
        y_pred_comb = combustion_model.predict(X_test_comb) 
        sc_test = combustion_model.score(X_test_comb, y_test_comb)
        rmse_te = np.sqrt(mean_squared_error(y_pred_comb, y_test_comb))
        mae= mean_absolute_error(y_pred_comb, y_test_comb)
        r2 = r2_score(y_pred_comb, y_test_comb)
        st.write(" XGBRegressor score: ", sc_test)
        st.write(" RMSE: ", rmse_te)
        st.write(" R2: ", r2)
     
        
             
          
    # fuel_rate = st.number_input("Fuel Rate (L/h)", value=5.0)
    # efficiency = st.number_input("Efficiency (%)", value=30.0)

    # # Prepare input for prediction
    # preselection_data = [[fuel_rate, efficiency]]

    # # Make prediction
    # energy_output_prediction = combustion_model.predict(preselection_data)[0]
    # st.write(f"Predicted Energy Output: {energy_output_prediction} kWh")
