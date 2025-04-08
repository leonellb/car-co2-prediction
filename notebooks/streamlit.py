import streamlit as st

# Set up the main title of the application
st.title('Predicting Vehicle CO₂ Emissions')

# Set up sidebar navigation
page = st.sidebar.selectbox("Choose a page:", ["Home", "Combustion", "Electric"])

if page == "Home":
    st.header("Welcome to the CO2 Project Defense Presentation")
    st.write("Here, we will present you with our different model prediction results: Please select 'Combustion' or 'Electric' from the sidebar to view specific analyses.")

elif page == "Combustion":
    st.header("Combustion Cars Analysis")
    st.write("Welcome to the Combustion side of our modeling project. Below, we will provide you with our prediction results for the selected models:")

    #@Richard: here, paste your code to import the combustion csv and do the preprocessing steps to prepare the data_test variable.

    use_case = st.radio("Select Data Type", ("Prediction with Unsmoted Data Training Data", "Prediction with Smoted Data Training Data"))
    if use_case == "Prediction with Unsmoted Data Training":
        st.write("You have selected Unsmoted Data Training for Combustion Cars.")
        # Add more interactive elements or outputs specific to this choice
        #@Richard the "Smote" thing is just a placeholder. Here you can import your model file and predict on the data_test variable you prepared above
        #You can insert more sub-parts here (as I prepared with "unsmoted" and "smoted" as an example) if you want more selection options (not necessary, just depending on your needs)
    else:
        st.write("You have selected Smoted Data Training for Combustion Cars.")
        # Add more interactive elements or outputs specific to this choice

elif page == "Electric":
    st.header("Electric Cars Analysis")
    st.write("Welcome to the Electric side of our modeling project. Below, we will provide you with our prediction results for the selected models:")

    #@Philipp/Leonel: Feel Free

    use_case = st.selectbox("Select Data Type", ["Prediction with Unsmoted Data Training", "Prediction with Smoted Data Training"])
    if use_case == "Prediction with Unsmoted Data Training":
        st.write("You have selected Unsmoted Data Training for Electric Cars.")
        # Add more interactive elements or outputs specific to this choice
    else:
        st.write("You have selected Smoted Data Training for Electric Cars.")
        # Add more interactive elements or outputs specific to this choice

#expand the app with more results as needed
