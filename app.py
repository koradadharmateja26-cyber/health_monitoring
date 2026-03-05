import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv


# 1. GLOBAL SETTINGS (Only one allowed per app)
st.set_page_config(page_title="Integrated Health System", layout="wide", page_icon="🏥")

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
    # Paste everything from INSIDE the main() function of your ap1.py here
    st.title("Diabetes Prediction App")
    # ... (Add input fields and prediction logic from ap1.py)
    try:
    model = pickle.load(open('trained_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Model or Scaler files not found! Please run the saving cells first.")

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the following details to check the diabetes status:")

    # 2. UI Layout
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

    # 3. Prediction Logic
    if st.button("Predict Result"):
        user_input = [float(preg), float(gluc), float(bp), float(skin),
                      float(ins), float(bmi), float(dpf), float(age)]

        input_data = np.asarray(user_input).reshape(1,-1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        st.markdown("---") # Visual separator

        if prediction[0] == 1:
            st.error("### Result: Positive")
            st.write("The person is likely Diabetic. Please consult a doctor for a professional diagnosis.")

            # --- PRECAUTIONS & DIET SECTION ---
            st.subheader("📋 Recommended Precautions & Diet")

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

if __name__ == '__main__':
    main()


# --- MENU 2: HEART DISEASE ---
elif selected == "Heart Disease":
    # Paste everything from INSIDE the main() function of your heart.py here
    st.title("❤️ Heart Disease Prediction App")
    # ... (Add input fields and prediction logic from heart.py)
    # 1. Page Configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# 2. Load the Heart Disease model and scaler with path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'heart_model.sav')
scaler_path = os.path.join(BASE_DIR, 'heart_scaler.sav')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    ready = True
except FileNotFoundError:
    st.error(f"Files not found! Ensure 'heart_model.sav' and 'heart_scaler.sav' are in: {BASE_DIR}")
    ready = False

def main():
    st.title("❤️ Heart Disease Prediction App")

    if not ready:
        return

    st.write("Enter the following clinical data to predict heart disease risk:")

    # 3. UI Layout - Organized into 2 columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=25)
        sex = st.selectbox('Sex (1 = Male, 0 = Female)', options=[1, 0])
        cp = st.selectbox('Chest Pain Type (0, 1, 2, 3)', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', value=120)
        chol = st.number_input('Serum Cholestoral (mg/dl)', value=200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', options=[1, 0])

    with col2:
        restecg = st.selectbox('Resting ECG Results (0, 1, 2)', options=[0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', value=150)
        exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', options=[1, 0])
        oldpeak = st.number_input('ST depression (Oldpeak)', value=0.0, step=0.1)
        slope = st.selectbox('Slope of the Peak Exercise ST Segment (0, 1, 2)', options=[0, 1, 2])
        ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
        thal = st.selectbox('Thal (0 = Normal, 1 = Fixed, 2 = Reversable)', options=[0, 1, 2])

    # 4. Prediction Logic
    if st.button("Predict Heart Condition"):
        try:
            features = [age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]

            input_data = np.asarray(features).reshape(1, -1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("### Result: High Risk of Heart Disease")
                
                # --- PRECAUTIONS & DIET SECTION ---
                st.subheader("📋 Recommended Precautions & Heart-Healthy Diet")
                
                d_col1, d_col2 = st.columns(2)
                
                with d_col1:
                    st.markdown("""
                    **Dietary Recommendations:**
                    * **Reduce Sodium:** Limit salt to help manage blood pressure.
                    * **Healthy Fats:** Choose olive oil, avocados, and nuts over butter/trans fats.
                    * **Omega-3:** Increase intake of fatty fish (like salmon) or flaxseeds.
                    * **High Fiber:** Eat more whole grains, fruits, and vegetables.
                    * **Limit Red Meat:** Opt for lean proteins like chicken or plant-based proteins.
                    """)
                
                with d_col2:
                    st.markdown("""
                    **Lifestyle Precautions:**
                    * **Regular Exercise:** Aim for 30 mins of moderate activity daily.
                    * **Quit Smoking:** Smoking is a leading cause of heart disease.
                    * **Stress Management:** Practice yoga or meditation.
                    * **Monitor BP:** Keep track of your blood pressure regularly.
                    """)

                st.markdown("---")
                st.subheader("🎥 Educational Resources (YouTube)")
                yt_col1, yt_col2, yt_col3 = st.columns(3)
                
                with yt_col1:
                    st.video("https://www.youtube.com/watch?v=njv_fC_O0I0")
                    st.caption("Understanding Heart Disease")
                
                with yt_col2:
                    st.video("https://www.youtube.com/watch?v=fXm0S8p7y7o")
                    st.caption("Top 10 Heart-Healthy Foods")
                
                with yt_col3:
                    st.video("https://www.youtube.com/watch?v=v8V5E6A7L7E")
                    st.caption("Exercises for a Stronger Heart")

                st.warning("🚨 **Medical Disclaimer:** This app is for educational purposes. Please consult a Cardiologist for a professional diagnosis.")

            else:
                st.success("### Result: Low Risk of Heart Disease")
                st.balloons()
                st.write("Your results suggest a lower risk. Continue maintaining a healthy lifestyle and balanced diet!")

        except Exception as e:
            st.warning(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()

# --- MENU 3: PARKINSON'S ---
elif selected == "Parkinson's":
    # Paste everything from INSIDE the main() function of your parkinsons.py here
    st.title("🧠 Parkinson's Disease Prediction App")
    # ... (Add input fields and prediction logic from parkinsons.py)
    # 1. Page Configuration
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")

# 2. Load the Parkinson's model and scaler with path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'parkinsons_model.sav')
scaler_path = os.path.join(BASE_DIR, 'parkinsons_scaler.sav')

try:
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    ready = True
except FileNotFoundError:
    st.error("Parkinson's Model or Scaler files not found! Please check filenames in your GitHub repo.")
    ready = False

def main():
    st.title("🧠 Parkinson's Disease Prediction App")
    st.write("This tool uses voice acoustic parameters (MDVP) to predict the likelihood of Parkinson's Disease.")

    if not ready:
        return

    # 3. UI Layout - 3 Columns for 22 features
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

    # 4. Prediction Logic
    if st.button("Predict Parkinson's Status"):
        try:
            features = [fo, fhi, flo, jitter_p, jitter_abs, rap, ppq, ddp,
                        shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                        rpde, dfa, spread1, spread2, d2, ppe]
            
            # Standardize and Predict
            input_data = np.asarray(features).reshape(1,-1)
            std_data = scaler.transform(input_data)
            prediction = model.predict(std_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("### Result: The model predicts a high likelihood of Parkinson's Disease.")
                
                # --- PRECAUTIONS & DIET SECTION ---
                st.subheader("📋 Management, Precautions & Diet")
                p_col1, p_col2 = st.columns(2)
                
                with p_col1:
                    st.markdown("""
                    **Dietary Suggestions:**
                    * **Antioxidants:** Eat plenty of fruits (berries) and vegetables (spinach, kale).
                    * **Omega-3 Fatty Acids:** Include walnuts, flaxseeds, and fish to support brain health.
                    * **Fiber:** Increase fiber intake to help with digestive issues common in Parkinson's.
                    * **Hydration:** Drink at least 8 glasses of water daily.
                    * **Protein Timing:** If on Levodopa, consult a doctor about taking protein at specific times.
                    """)
                
                with p_col2:
                    st.markdown("""
                    **Daily Precautions:**
                    * **Physical Therapy:** Regular stretching and balance exercises are vital.
                    * **Speech Therapy:** Practice vocal exercises to maintain voice strength.
                    * **Fall Prevention:** Ensure the home is well-lit and free of tripping hazards.
                    * **Mental Health:** Stay socially active and engage in hobbies to support mood.
                    """)

                st.markdown("---")
                st.subheader("🎥 Helpful YouTube Resources")
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    st.video("https://www.youtube.com/watch?v=ARdGsh_D_XU")
                    st.caption("Understanding Parkinson's Symptoms")
                
                with v_col2:
                    st.video("https://www.youtube.com/watch?v=S_3f760x0E0")
                    st.caption("Exercises for Parkinson's Patients")

                st.warning("🚨 **Disclaimer:** This is an AI-based prediction tool and not a medical diagnosis. Please see a Neurologist for professional evaluation.")

            else:
                st.success("### Result: The model predicts the person does NOT have Parkinson's Disease.")
                st.balloons()
                st.write("Continue maintaining your health with regular checkups and a balanced lifestyle.")

        except Exception as e:
            st.warning(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

# --- MENU 4: PRESCRIPTION DECODER ---
elif selected == "Prescription Decoder":
    # Paste the analyze_prescription function and main() logic from pra.py here
    st.title("🩺 AI Prescription Decoder")
    # Load environment variables
load_dotenv()

# Configure Gemini
# API_KEY = os.getenv("GEMINI_API_KEY")
# if API_KEY:
#     genai.configure(api_key=API_KEY)
# else:
#     st.error("Gemini API Key not found. Please set it in your environment variables.")

st.set_page_config(
    page_title="AI Prescription Decoder",
    page_icon="💊",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .header-text {
        color: #1a2a6c;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

def analyze_prescription(image):
    """
    Analyzes the prescription image using available Gemini models with fallback.
    """
    # Priority list of models to try
    models_to_try = ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-flash-latest']
    
    last_error = ""
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            
            prompt = """
            You are a highly skilled medical pharmacist. 
            Analyze the provided image of a handwritten prescription.
            1. Transcribe the handwriting accurately.
            2. Extract the following details for each medicine:
               - Medicine Name
               - Dosage (e.g., 500mg, 10ml)
               - Frequency (e.g., Twice a day, 1-0-1)
               - Duration (e.g., 5 days)
               - Special Instructions (e.g., Before food, avoid dairy)
            3. Provide a brief summary of what the prescription is for (if discernible).
            4. ADD A STRONG DISCLAIMER: "This is an AI-generated interpretation. Please verify with a qualified pharmacist or doctor."
            
            Format the output in a clean, structured way using Markdown tables.
            """
            
            response = model.generate_content([prompt, image])
            return f"*(Using model: {model_name})*\n\n" + response.text
            
        except Exception as e:
            last_error = str(e)
            # If it's a quota error or 404, try the next model
            if "429" in last_error or "404" in last_error:
                continue
            else:
                return f"Error during analysis: {last_error}"
    
    return f"Failed to analyze. Last error: {last_error}\n\nAll tried models: {models_to_try}"

def main():
    st.markdown("<h1 class='header-text'>🩺 AI Prescription Decoder</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Transform unreadable handwriting into clear medical guidance.</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Simple sidebar for API key if not in Env
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("API Key Configured!")
        else:
            st.warning("Please enter your Gemini API Key to proceed.")

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📤 Upload Prescription")
        uploaded_file = st.file_uploader("Choose an image of your prescription...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Prescription"):
                if not api_key:
                    st.error("Please provide an API key in the sidebar first.")
                else:
                    with st.spinner("🧠 AI is decoding the handwriting..."):
                        analysis_result = analyze_prescription(image)
                        st.session_state['analysis_result'] = analysis_result
    
    with col2:
        st.subheader("📋 Clear Interpretation")
        if 'analysis_result' in st.session_state:
            st.markdown(st.session_state['analysis_result'])
        else:
            st.info("Upload and analyze a prescription to see the digital interpretation here.")

if __name__ == "__main__":
    main()
