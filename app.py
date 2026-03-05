import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# 1. GLOBAL SETTINGS (Must be at the very top and only once)
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

# --- HELPER FUNCTION FOR DECODER ---
def analyze_prescription(image):
    models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest']
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = """
            You are a highly skilled medical pharmacist. 
            Analyze the provided image of a handwritten prescription.
            1. Transcribe the handwriting accurately.
            2. Extract details: Medicine Name, Dosage, Frequency, Duration, Instructions.
            3. Provide a brief summary and a STRONG DISCLAIMER.
            Format the output in Markdown tables.
            """
            response = model.generate_content([prompt, image])
            return f"*(Using model: {model_name})*\n\n" + response.text
        except Exception as e:
            continue
    return "Failed to analyze with available models."

# --- MAIN APP LOGIC ---

# MENU 1: DIABETES (Your ap1.py content)
if selected == "Diabetes":
    st.title("Diabetes Prediction")
    st.write("Enter the following details to check the diabetes status:")

    try:
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
                st.write("The person is likely Diabetic. Please consult a doctor for a professional diagnosis.")
                st.subheader("Recommended Precautions & Diet")
                col_diet, col_links = st.columns(2)
                with col_diet:
                    st.markdown("""
                    **Dietary Tips:**
                    * **Eat more Fiber:** Focus on whole grains, beans, and leafy greens.
                    * **Reduce Sugars:** Avoid sodas, candies, and processed snacks.
                    * **Portion Control:** Use smaller plates to manage calorie intake.
                    * **Healthy Fats:** Opt for nuts, seeds, and olive oil.
                    """)
                with col_links:
                    st.markdown("""
                    **Educational Resources:**
                    * [Understanding Diabetes (Mayo Clinic)](https://www.youtube.com/watch?v=X9ivR4y03DE)
                    * [Best Foods for Diabetics](https://www.youtube.com/watch?v=PrUu8V5A1iI)
                    * [Exercise Tips for Blood Sugar](https://www.youtube.com/watch?v=X_pU6H7P01Q)
                    """)
            else:
                st.success("### Result: Negative")
                st.write("The person is likely Not Diabetic. Continue maintaining a healthy lifestyle!")

# MENU 2: HEART DISEASE (Your heart.py content)
elif selected == "Heart Disease":
    st.title("Heart Disease Prediction")
    try:
        model_h = pickle.load(open(os.path.join(BASE_DIR, 'heart_model.sav'), 'rb'))
        scaler_h = pickle.load(open(os.path.join(BASE_DIR, 'heart_scaler.sav'), 'rb'))
        ready_h = True
    except FileNotFoundError:
        st.error("Heart Model files not found!")
        ready_h = False

    if ready_h:
        col1, col2 = st.columns(2)
        with col1:
            h_age = st.number_input('Age', min_value=1, max_value=120, value=25)
            sex = st.selectbox('Sex (1 = Male, 0 = Female)', options=[1, 0])
            cp = st.selectbox('Chest Pain Type (0, 1, 2, 3)', options=[0, 1, 2, 3])
            trestbps = st.number_input('Resting Blood Pressure', value=120)
            chol = st.number_input('Serum Cholestoral', value=200)
            fbs = st.selectbox('Fasting Blood Sugar > 120', options=[1, 0])
        with col2:
            restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])
            thalach = st.number_input('Max Heart Rate Achieved', value=150)
            exang = st.selectbox('Exercise Induced Angina', options=[1, 0])
            oldpeak = st.number_input('ST depression', value=0.0)
            slope = st.selectbox('Slope', options=[0, 1, 2])
            ca = st.selectbox('Major Vessels', options=[0, 1, 2, 3])
            thal = st.selectbox('Thal', options=[0, 1, 2])

        if st.button("Predict Heart Condition"):
            features = [h_age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            std_data = scaler_h.transform(np.asarray(features).reshape(1, -1))
            prediction = model_h.predict(std_data)
            
            if prediction[0] == 1:
                st.error("### Result: High Risk of Heart Disease")
                st.subheader("Educational Resources (YouTube)")
                yt_col1, yt_col2, yt_col3 = st.columns(3)
                with yt_col1: st.video("https://www.youtube.com/watch?v=njv_fC_O0I0")
                with yt_col2: st.video("https://www.youtube.com/watch?v=fXm0S8p7y7o")
                with yt_col3: st.video("https://www.youtube.com/watch?v=v8V5E6A7L7E")
            else:
                st.success("### Result: Low Risk")

# --- MENU 3: PARKINSON'S DISEASE (Your parkinsons.py content) ---
elif selected == "Parkinson's":
    st.title("Parkinson's Disease Prediction")
    st.write("This tool uses voice acoustic parameters (MDVP) to predict the likelihood of Parkinson's Disease.")

    # 1. Load the Parkinson's model and scaler
    try:
        model_p = pickle.load(open(os.path.join(BASE_DIR, 'parkinsons_model.sav'), 'rb'))
        scaler_p = pickle.load(open(os.path.join(BASE_DIR, 'parkinsons_scaler.sav'), 'rb'))
        ready_p = True
    except FileNotFoundError:
        st.error("Parkinson's Model or Scaler files not found! Check your GitHub repo.")
        ready_p = False

    if ready_p:
        # 2. UI Layout - 3 Columns for all 22 features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', value=0.0, format="%.3f")
            fhi = st.number_input('MDVP:Fhi(Hz)', value=0.0, format="%.3f")
            flo = st.number_input('MDVP:Flo(Hz)', value=0.0, format="%.3f")
            jitter_p = st.number_input('MDVP:Jitter(%)', value=0.0, format="%.5f")
            jitter_abs = st.number_input('MDVP:Jitter(Abs)', value=0.0, format="%.5f")
            rap = st.number_input('MDVP:RAP', value=0.0, format="%.5f")
            ppq = st.number_input('MDVP:PPQ', value=0.0, format="%.5f")
            
        with col2:
            ddp = st.number_input('Jitter:DDP', value=0.0, format="%.5f")
            shimmer = st.number_input('MDVP:Shimmer', value=0.0, format="%.5f")
            shimmer_db = st.number_input('MDVP:Shimmer(dB)', value=0.0, format="%.3f")
            apq3 = st.number_input('Shimmer:APQ3', value=0.0, format="%.5f")
            apq5 = st.number_input('Shimmer:APQ5', value=0.0, format="%.5f")
            apq = st.number_input('MDVP:APQ', value=0.0, format="%.5f")
            dda = st.number_input('Shimmer:DDA', value=0.0, format="%.5f")
            
        with col3:
            nhr = st.number_input('NHR', value=0.0, format="%.5f")
            hnr = st.number_input('HNR', value=0.0, format="%.3f")
            rpde = st.number_input('RPDE', value=0.0, format="%.5f")
            dfa = st.number_input('DFA', value=0.0, format="%.5f")
            spread1 = st.number_input('spread1', value=0.0, format="%.5f")
            spread2 = st.number_input('spread2', value=0.0, format="%.5f")
            d2 = st.number_input('D2', value=0.0, format="%.5f")
            ppe = st.number_input('PPE', value=0.0, format="%.5f")

        # 3. Prediction Logic
        if st.button("Predict Parkinson's Status"):
            try:
                # Grouping all 22 inputs for the model
                features = [fo, fhi, flo, jitter_p, jitter_abs, rap, ppq, ddp,
                            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                            rpde, dfa, spread1, spread2, d2, ppe]
                
                # Standardize and Predict
                input_data = np.asarray(features).reshape(1,-1)
                std_data = scaler_p.transform(input_data)
                prediction = model_p.predict(std_data)

                st.markdown("---")
                if prediction[0] == 1:
                    st.error("### Result: High likelihood of Parkinson's Disease.")
                    
                    # --- YOUR PRECAUTIONS & DIET SECTION ---
                    st.subheader("📋 Management, Precautions & Diet")
                    p_col1, p_col2 = st.columns(2)
                    
                    with p_col1:
                        st.markdown("""
                        **Dietary Suggestions:**
                        * **Antioxidants:** Berries and leafy greens.
                        * **Omega-3:** Walnuts and fish for brain health.
                        * **Hydration:** Drink at least 8 glasses of water.
                        """)
                    
                    with p_col2:
                        st.markdown("""
                        **Daily Precautions:**
                        * **Physical Therapy:** Stretching and balance exercises.
                        * **Speech Therapy:** Practice vocal exercises.
                        * **Fall Prevention:** Keep rooms well-lit.
                        """)

                    # --- YOUR VIDEO LINKS ---
                    st.markdown("---")
                    st.subheader("Helpful YouTube Resources")
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.video("https://www.youtube.com/watch?v=ARdGsh_D_XU")
                        st.caption("Symptoms Guide")
                    with v_col2:
                        st.video("https://www.youtube.com/watch?v=S_3f760x0E0")
                        st.caption("Exercise Routine")
                else:
                    st.success("### Result: The model predicts NO Parkinson's Disease.")
                    st.balloons()
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# MENU 4: PRESCRIPTION DECODER (Your pra.py content)
elif selected == "Prescription Decoder":
    st.markdown("<h1 style='text-align: center;'>Prescription Decoder</h1>", unsafe_allow_html=True)
    with st.sidebar:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key: genai.configure(api_key=api_key)
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=400)
        if st.button("Analyze Prescription"):
            if not api_key: st.error("Enter API Key")
            else:
                with st.spinner("Analyzing..."):
                    res = analyze_prescription(image)
                    st.markdown(res)



