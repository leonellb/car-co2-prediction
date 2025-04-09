import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

import streamlit as st


@st.cache_data
def joblib_load(input_file):
    return joblib.load(input_file)


@st.cache_data
def model_predict(model, data):
    return model.predict(data)


# Set up the main title of the application
st.title("Predicting Vehicle CO₂ Emissions")

# Set up sidebar navigation
page = st.sidebar.selectbox("Choose a page:", ["Home", "Combustion", "Electric"])

if page == "Home":
    st.header("Welcome to the CO2 Project Defense Presentation")
    st.write("Here, we will present you with our different model prediction results: Please select 'Combustion' or 'Electric' from the sidebar to view specific analyses.")

elif page == "Combustion":
    st.header("Combustion Cars Analysis")
    st.write("Welcome to the Combustion side of our modeling project. Below, we will provide you with our prediction results for the selected models:")

    X_train_comb = joblib_load("/content/drive/My Drive/X_train_comb.pkl")
    X_test_comb = joblib_load("/content/drive/My Drive/X_test_comb.pkl")
    y_train_comb = joblib_load("/content/drive/My Drive/y_train_comb.pkl")
    y_test_comb = joblib_load("/content/drive/My Drive/y_test_comb.pkl")

    XGB_model = joblib_load("/content/drive/My Drive/modelXGBRegressor.pkl")

    # score of XGBRegressor
    pred_train = model_predict(XGB_model, X_train_comb)
    pred_test = model_predict(XGB_model, X_test_comb)
    sc_train = XGB_model.score(X_train_comb, y_train_comb)
    sc_test = XGB_model.score(X_test_comb, y_test_comb)
    st.write("score: ", sc_train)
    st.write("score: ", sc_test)

    # root-mean-squared-error
    y_pred_train = model_predict(XGB_model, X_train_comb)
    y_pred_test = model_predict(XGB_model, X_test_comb)
    rmse_tr = np.sqrt(mean_squared_error(y_pred_train, y_train_comb))
    rmse_te = np.sqrt(mean_squared_error(y_pred_test, y_test_comb))
    st.write("rmse training data: ", rmse_tr)
    st.write("rmse test data: ", rmse_te)

elif page == "Electric":
    st.header("Electric Cars Analysis")
    st.write("Welcome to the Electric side of our modeling project. Below, we will provide you with our prediction results for the selected models:")

    # @Philipp/Leonel: Feel Free

    use_case = st.selectbox("Select Data Type", ["Prediction with Unsmoted Data Training", "Prediction with Smoted Data Training"])
    if use_case == "Prediction with Unsmoted Data Training":
        st.write("You have selected Unsmoted Data Training for Electric Cars.")
        # Add more interactive elements or outputs specific to this choice
    else:
        st.write("You have selected Smoted Data Training for Electric Cars.")
        # Add more interactive elements or outputs specific to this choice

# expand the app with more results as needed
