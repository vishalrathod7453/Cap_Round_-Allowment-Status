import streamlit as st
import pickle
import numpy as np

# Set up page layout and configurations
st.set_page_config(
    page_title="EduMatch | College Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Attractive UI Design, Animations, and Frosted Glass Layouts via CSS
st.markdown("""
    <style>
    /* Full page soft gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Elegant Glowing Header Banner */
    .header-box {
        background: rgba(255, 255, 255, 0.04);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        margin-bottom: 2.5rem;
        animation: fadeInDown 1.2s ease-out;
    }
    
    .header-box h1 {
        background: linear-gradient(90deg, #38bdf8, #c084fc, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-box p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Floating Card Layout for Data Inputs */
    .input-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        box-shadow: 0 30px 60px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Dynamic Pulse Animation for Predicted Results Card */
    .result-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(192, 132, 252, 0.15) 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed #c084fc;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 0 30px rgba(192, 132, 252, 0.3);
        animation: pulseGlow 2.5s infinite;
    }
    
    .result-title {
        color: #67e8f9;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .result-college {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(255,255,255,0.2);
    }
    
    /* CSS Animation Definitions */
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 20px rgba(192, 132, 252, 0.2); border-color: #c084fc; }
        50% { box-shadow: 0 0 40px rgba(56, 189, 248, 0.5); border-color: #38bdf8; }
        100% { box-shadow: 0 0 20px rgba(192, 132, 252, 0.2); border-color: #c084fc; }
    }
    
    /* Styled labels for widgets */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    
    /* Style refinement for Streamlit components */
    .stNumberInput div div input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allowed=True)

# Main Banner Area
st.markdown("""
    <div class="header-box">
        <h1>🎓 EduMatch AI</h1>
        <p>Intelligent College Predictor Engine powered by Machine Learning</p>
    </div>
""", unsafe_allowed=True)

# Helper function to safely read the pickled pipeline model
@st.cache_resource
def load_prediction_model():
    try:
        with open("Modelclg.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

model = load_prediction_model()

# Form Input Container Card
st.markdown('<div class="input-card">', unsafe_allowed=True)
st.subheader("📊 Enter Academic Metrics")

# Creating inputs for features
# Adjust min_value, max_value, and step values according to your data distribution
feature_1 = st.number_input("🎯 entrance Percentile / Cut-off Score", min_value=0.0, max_value=100.0, value=95.5, step=0.01)
feature_2 = st.number_input("📝 Secondary Exam Score / Intermediate Mark (%)", min_value=0.0, max_value=100.0, value=88.0, step=0.1)

st.markdown('</div>', unsafe_allowed=True)

# Action Area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Large responsive submit button
    trigger_prediction = st.button("✨ Find My Matching Campus", use_container_width=True)

# Execution block following form submission
if trigger_prediction:
    if model is not None:
        with st.spinner("Processing campus matching vectors..."):
            # Reshape inputs to a standard 2D array format for scikit-learn models
            input_vector = np.array([[feature_1, feature_2]])
            
            try:
                predicted_output = model.predict(input_vector)
                target_college = predicted_output[0]
                
                # Show celebratory effect on a successful match
                st.balloons()
                
                # Render predicted target beautifully
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">🌟 Ideal Institutional Match Found 🌟</div>
                        <div class="result-college">{target_college}</div>
                    </div>
                """, unsafe_allowed=True)
                
            except Exception as exception_log:
                st.error("Prediction failed. Please check that input dimensions match feature expectations.")
                st.caption(f"Error Log: {exception_log}")
    else:
        st.error("Prediction engine offline. Make sure 'Modelclg.pkl' exists in the active directory.")
