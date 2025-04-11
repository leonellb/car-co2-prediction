# General
def raw_csv_to_proc_csv_converter(file_name: str):
    """Simple method to share common renaming logic for raw to preprocessed csv files"""
    return file_name.replace(".csv", "_proc.csv")


# Preprocessing
# folder for files that are output only (e.g. plots, reports, etc.)
OUTPUT_DIR = "files/output/"
# folder for files that are used as input and output
FILES_DIR = "files/"

PRE_ANALYSIS_FILE = "files/23_22_21-eea_europa_eu-CarsCO2.csv"
RAW_DATA_FILES = ["files/23_22_21-eea_europa_eu-CarsCO2.csv"]

# preprocessed files are only accessible if you executed 1_1-prep_database_file_generator.ipynb
PREP_DATA_FILES = [raw_csv_to_proc_csv_converter(file) for file in RAW_DATA_FILES]

MERGED_DATA_FILE = "files/23_22_21-eea_europa_eu-CarsCO2_proc.csv"
MERGED_COMBUSTION_FILE = "files/23_22_21-eea_europa_eu-CarsCO2_combustion.csv"
MERGED_ELECTRIC_FILE = "files/23_22_21-eea_europa_eu-CarsCO2_electric.csv"

ENCODED_PREP_ELECTRIC_FILE = "files/23_22_21-eea_europa_eu-CarsCO2_electric_enc.csv"

DENSITY_THRESHOLD = 0.7
UNIQUE_VALUES_THRESHOLD = 30

# columns to be dropped without further analysis required (mostly related to results of 1_0-prep_pre_analysis.ipynb)
COLS_PRE_DROP = ["Mp", "VFN", "Man", "Mk", "MMS", "Tan", "Va", "Ve", "Cr", "Enedc (g/km)", "W (mm)", "At1 (mm)", "At2 (mm)", "Ernedc (g/km)", "De", "Vf", "r", "Status", "Date of registration", "RLFI", "ech"]

REPLACE_STRING_OTHER = "other"

# update this mapper to select interesting columns including rename to proper format
COLS_MAPPER = {
    "Cn": "commercial_name",
    "Ct": "category_of_vehicle",  # EU Fahrzeugklasse
    "Country": "member_state",
    "ec (cm3)": "engine_capacity",
    "ep (KW)": "engine_power",
    "Erwltp (g/km)": "erwltp",
    "Electric range (km)": "electric_range",
    "Ewltp (g/km)": "specific_co2_emissions",
    "Fm": "fuel_mode",
    "Ft": "fuel_type",
    "Fuel consumption ": "fuel_consumption",
    "IT": "innovative_technologies",
    "m (kg)": "mass_vehicle",
    "Mh": "manufacturer_name_eu",
    "Mt": "weltp_test_mass",
    "T": "vehicle_type",
    "z (Wh/km)": "electric_energy_consumption",
}

DATABASE_FILE_INDEX = "ID"
DATABASE_FILE_DTYPES = {
    "member_state": "object",
    "manufacturer_name_eu": "object",
    "vehicle_type": "object",
    "commercial_name": "object",
    "category_of_vehicle": "object",
    "fuel_type": "object",
    "fuel_mode": "object",
    "innovative_technologies": "object",
    "mass_vehicle": "float64",
    "weltp_test_mass": "float64",
    "engine_capacity": "float64",
    "engine_power": "float64",
    "erwltp": "float64",
    "year": "int64",
    "electric_range": "float64",
    "electric_energy_consumption": "float64",
    "fuel_consumption": "float64",
    "specific_co2_emissions": "float64",
}

# columns_to_analyse for df_combustion and for df_electric
# MAKE SURE to update fuel_types if other datasets have types we did not consider.
COMBUSTION_FUEL_TYPES = ["diesel", "petrol", "petrol/electric", "ng", "lpg", "ng-biomethane", "e85", "diesel/electric"]
ELECTRIC_FUEL_TYPES = ["electric"]
ELECTRIC_TARGET = "electric_energy_consumption"
# Model Training
TRAIN_TEST_SPLIT_ELECTRIC_FILE = "files/electric-train_test_split.pkl"
SCALER_ELECTRIC_FILE = "files/electric-scaler.pk1"

TRAIN_TEST_SPLIT_ELECTRIC_FILE_SAMPLED = "files/electric-train_test_split_sampled.pkl"
SCALER_ELECTRIC_FILE_SAMPLED = "files/electric-scaler_sampled.pk1"

# electric model files
ELECTRIC_MODEL_FILE = "files/output/models/electric-model.pkl"
ELECTRIC_NN_MODEL_FILE = "files/output/models/electric-nn_model.keras"
ELECTRIC_NN_CKPT_MODEL_FILE = "files/output/models/electric-nn_model_ckpt.keras"
# store with additional epoch information
# ELECTRIC_NN_CKPT_MODEL_FILE = "files/output/models/electric-nn_model_ckpt-{epoch:04d}.keras"
ELECTRIC_NN_MODEL_FILE_USAMPLED = "files/output/models/electric-nn_model-usampled.keras"
ELECTRIC_NN_CKPT_MODEL_FILE_USAMPLED = "files/output/models/electric-nn_model_ckpt-usampled.keras"

# combustion model files
COMBUSTION_MODEL_FILE = "files/output/models/combustion-model.pkl"

# Streamlit presentation
ELECTRIC_COLS_VALS_SELECTION_FILE = "files/output/electric-unique_vals_cat_cols.json"
ELECTRIC_PREDICTION_PRESELECT_FILE = "files/output/electric-prediction-preselect.json"
