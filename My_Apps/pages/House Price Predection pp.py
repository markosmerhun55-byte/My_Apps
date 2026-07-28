import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Page config MUST be the very first Streamlit call executed
st.set_page_config(page_title="House Price", layout="centered")

# py -m streamlit run House_App.py
# Class definition needed if Custom Perceptron won
class LinearPerceptronRegressor:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

# Load Pipeline Assets
@st.cache_resource
def load_assets():
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "house_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

pipeline = load_assets()
model = pipeline["model"]
encoder = pipeline["encoder"]
scaler = pipeline["scaler"]
enc_type = pipeline["enc_type"]
cat_cols = pipeline["categorical_cols"]
num_cols = pipeline["numerical_cols"]
#CSS
st.markdown("""
    <style>

    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* Apply Font */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* 1. Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #f4f7fb 0%, #dbeafe 50%, #eef6ff 100%);
        color: #1e293b;
    }

    /* 2. Style Input Cards & Container Containers */
    div[data-testid="stForm"], div[data-testid="stBlock"] {
        border-radius: 12px;
    }

    /* 3. Header Colors */
    h1 {
        color: #1d4ed8 !important;
        font-weight: 700;
    }

    h2 {
        color: #2563eb !important;
        font-weight: 700;
    }

    h3 {
        color: #0f766e !important;
        font-weight: 700;
    }

    /* 4. Style Labels and Input Boxes */
    label {
        color: #334155 !important;
        font-weight: 700 !important;
    }

    /* Input Fields Border and Background */
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 8px !important;
        background-color: #ffffff !important;
        border: 1px solid #60a5fa !important;
    }

    /* Text Color inside Inputs */
    input {
        color: #1e293b !important;
    }

    /* 5. Predict Button Styling */
    div.stButton > button {
        width: 200%;
        background-color: #2563eb;
        color: white;
        font-size: 30px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 0;
        border: none;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #38bdf8;
        color: white;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }

    </style>
""", unsafe_allow_html=True)
# App Title & UI Header

st.title(":rainbow[House Price Prediction Web App]")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Input House Features</h1>", 
    unsafe_allow_html=True
)
user_inputs = {}
# Numeric inputs
for col in num_cols:
    user_inputs[col] = st.number_input(f"Enter {col}", value=0.0, step=1.0)

# Categorical inputs
for col in cat_cols:
    # If LabelEncoder, load stored classes directly from encoder
    if enc_type == "Label":
        options = list(encoder[col].classes_)
    else:
        # OneHotEncoder ColumnTransformer extraction
        for name, transformer, cols in encoder.transformers_:
            if name == "onehot" and col in cols:
                options = list(transformer.categories_[cols.index(col)])
                break
    user_inputs[col] = st.selectbox(f"Select {col}", options)

# Prediction Logic
if st.button("Get Price"):
    site_area = user_inputs.get("Site_Area_sqm", 0)
    built_area = user_inputs.get("Built_Area_sqm", 0)
    school_Dis= user_inputs.get("Proximity_to_Schools_km",0)
    bus_Dis= user_inputs.get("Proximity_to_Bus_Station_km",0)
    CBD= user_inputs.get("Proximity_to_CBD_km",0)
    Room= user_inputs.get("Number_of_Rooms",0)

    # Validation check:
    if Room < 1:
        st.error("⚠️ Validation Error: Number_of_Rooms must be greater than or equal to 1 ")
    elif site_area < 75: 
        st.error("⚠️ Validation Error: Site Area  must be greater than 75 sqm.")
    elif built_area < 75:
        st.error("⚠️ Validation Error: Built Area must be greater than 75 sqm.")
    elif CBD <=0:
        st.error("⚠️ Validation Error: Proximity_to_CBD_km must be greater than 0 sqm. ")
    elif bus_Dis <=0:
        st.error("⚠️ Validation Error: Proximity_to_Bus_Station_km must be greater than 0 sqm  ")
    elif school_Dis <=0:
        st.error("⚠️ Validation Error: Proximity_to_Schools_km must be greater than 0 sqm ")
    else:
        # Prepare inputs and proceed to prediction
        input_df = pd.DataFrame([user_inputs])

        if enc_type == "Label":
            encoded_df = input_df.copy()
            for col in cat_cols:
                encoded_df[col] = encoder[col].transform(encoded_df[col])
            processed_data = encoded_df.to_numpy()
        else:
            processed_data = encoder.transform(input_df)

        if scaler is not None:
            processed_data = np.asarray(processed_data)
            processed_data = scaler.transform(processed_data)

        # Make Prediction
        prediction = model.predict(processed_data)
        price = max(0, prediction[0])
        st.success(f"### House Price:{price} ETB")
        # Footer
st.divider()
st.caption("Powered by Mera")
