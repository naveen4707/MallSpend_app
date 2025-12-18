import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="🛍️",
    layout="centered"
)

# --- CUSTOM CSS FOR UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .low { background-color: #6c757d; }
    .average { background-color: #007bff; }
    .high { background-color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD THE MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('knn_mallspend.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# --- HEADER ---
st.title("🛍️ Mall Customer Spender Classifier")
st.write("Enter the customer's demographics to predict their spending behavior category.")
st.divider()

if model is None:
    st.error("⚠️ Model file 'knn_mall_model.pkl' not found. Please run the training script first.")
    st.stop()

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

with col2:
    income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=50)

# --- PREDICTION LOGIC ---
if st.button("Analyze Spending Potential"):
    # Prepare input for KNN
    features = np.array([[age, income]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Map result to Spend Categories
    st.divider()
    if prediction == 0:
        st.markdown('<div class="result-card low">📉 Low Spender</div>', unsafe_allow_html=True)
        st.info("This customer typically spends below average.")
    elif prediction == 1:
        st.markdown('<div class="result-card average">📊 Average Spender</div>', unsafe_allow_html=True)
        st.info("This customer has moderate spending habits.")
    else:
        st.markdown('<div class="result-card high">💰 High Spender!</div>', unsafe_allow_html=True)
        st.success("Target this customer! They have high spending potential.")

# --- FOOTER INFO ---
st.sidebar.markdown("### 🏬 About the Model")
st.sidebar.info("""
This app uses a **K-Nearest Neighbors (KNN)** model trained on Mall Customer data.
- **Features used:** Age & Annual Income.
- **Target:** Spending Group (based on spending scores).
""")
