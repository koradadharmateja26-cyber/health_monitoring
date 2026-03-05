import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 1. GLOBAL SETTINGS (MUST be the first Streamlit command)
st.set_page_config(page_title="Integrated Health System", layout="wide", page_icon="🏥")

# Load environment variables
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
    st.write("Enter the following details to check the diabetes status:")

    # Load Model and Scaler
    try:
        # Pushed to the right (indented) to fix your error
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
            dpf = st.text_input('Diabetes Pedigree Function', value="0")
            age = st.text_input('Age', value="0")

        if st.button("Predict Result"):
            user_input = [float(preg), float(gluc), float(bp), float(skin),
                          float(ins), float(bmi), float(dpf), float(age)]
            input_data = np.asarray(user_input).reshape(1,-1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("### Result: Positive")
                st.write("The person is likely Diabetic.")
                st.subheader("📋 Recommended Precautions & Diet")
                col_diet, col_links = st.columns(2)
                with col_diet:
                    st.markdown("**Dietary Tips:**\n* Eat more Fiber\n* Reduce Sugars\n* Portion Control")
                with col_links:
                    st.markdown("**Resources:**\n* [Diabetes Clinic](https://www.youtube.com/watch?v=X9ivR4y03DE)")
            else:
                st.success("### Result: Negative")
                st.write("The person is likely Not Diabetic.")

# --- MENU 2: HEART DISEASE ---
elif selected == "Heart Disease":
    st.title("❤️ Heart Disease Prediction App")
    
    model_path = os.path.join(BASE_DIR, 'heart_model.sav')
    scaler_path = os.path.join(BASE_DIR, 'heart_scaler.sav')

    try:
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        ready = True
    except FileNotFoundError:
        st.error(f"Heart files not found in: {BASE_DIR}")
        ready = False

    if ready:
        st.write("Enter clinical data to predict heart disease risk:")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=25)
            sex = st.selectbox('Sex (1 = Male, 0 = Female)', options=[1, 0])
            cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
            trestbps = st.number_input('Resting Blood Pressure', value=120)
            chol = st.number_input('Serum Cholestoral', value=200)
            fbs = st.selectbox('Fasting Blood Sugar > 120', options=[1, 0])
        with col2:
            restecg = st.selectbox('Resting ECG', options=[0, 1, 2])
            thalach = st.number_input('Max Heart Rate', value=150)
            exang = st.selectbox('Exercise Angina', options=[1, 0])
            oldpeak = st.number_input('ST depression', value=0.0, step=0.1)
            slope = st.selectbox('ST Segment Slope', options=[0, 1, 2])
            ca = st.selectbox('Major Vessels', options=[0, 1, 2, 3])
            thal = st.selectbox('Thal', options=[0, 1, 2])

        if st.button("Predict Heart Condition"):
            features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            input_data = np.asarray(features).reshape(1, -1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("### Result: High Risk")
                st.video("https://www.youtube.com/watch?v=njv_fC_O0I0")
            else:
                st.success("### Result: Low Risk")
                st.balloons()

# --- MENU 3: PARKINSON'S ---
elif selected == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction App")
    
    model_path = os.path.join(BASE_DIR, 'parkinsons_model.sav')
    scaler_path = os.path.join(BASE_DIR, 'parkinsons_scaler.sav')

    try:
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        ready = True
    except FileNotFoundError:
        st.error("Parkinson's Model files not found!")
        ready = False

    if ready:
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', value=0.0, format="%.3f")
            fhi = st.number_input('MDVP:Fhi(Hz)', value=0.0, format="%.3f")
            jitter_p = st.number_input('MDVP:Jitter(%)', value=0.0, format="%.5f")
        # ... (Simplified for brevity, add your other 19 inputs here)
        
        if st.button("Predict Parkinson's Status"):
            # Ensure all 22 features are in this list
            features = [fo, fhi, 0, jitter_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
            input_data = np.asarray(features).reshape(1,-1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)
            if prediction[0] == 1:
                st.error("High Likelihood of Parkinson's")
            else:
                st.success("Likely Negative")

# --- MENU 4: PRESCRIPTION DECODER ---
elif selected == "Prescription Decoder":
    st.title("🩺 AI Prescription Decoder")
    
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            genai.configure(api_key=api_key)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        uploaded_file = st.file_uploader("Upload prescription", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            if st.button("Analyze"):
                if not api_key:
                    st.error("Provide API Key")
                else:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(["Decode this medical prescription and provide a table.", image])
                    st.session_state['analysis_result'] = response.text
    
    with col2:
        if 'analysis_result' in st.session_state:
            st.markdown(st.session_state['analysis_result'])
