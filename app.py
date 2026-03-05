import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 1. GLOBAL SETTINGS (Must be the very first Streamlit command)
st.set_page_config(page_title="Integrated Health System", layout="wide", page_icon="🏥")

load_dotenv()

# 2. SHARED SIDEBAR MENU
with st.sidebar:
    st.title("Main Menu")
    selected = st.selectbox(
        "Choose Prediction System:",
        ["Diabetes", "Heart Disease", "Parkinson's", "Prescription Decoder"]
    )

# 3. GLOBAL PATH HANDLING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- MENU 1: DIABETES ---
if selected == "Diabetes":
    st.title("Diabetes Prediction App")
    
    try:
        # FIXED: These lines must be indented 4 spaces to be inside the try block
        model = pickle.load(open(os.path.join(BASE_DIR, 'trained_model.sav'), 'rb'))
        scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.sav'), 'rb'))
        ready = True
    except FileNotFoundError:
        st.error("Diabetes Model or Scaler files not found!")
        ready = False

    if ready:
        col1, col2 = st.columns(2)
        with col1:
            preg = st.text_input('Number of Pregnancies', value="0")
            gluc = st.text_input('Glucose Level', value="0")
            bp = st.text_input('Blood Pressure value', value="0")
            skin = st.text_input('Skin Thickness value', value="0")
        with col2:
            ins = st.text_input('Insulin Level', value="0")
            bmi = st.text_input('BMI value', value="0")
            dpf = st.text_input('Diabetes Pedigree Function',
