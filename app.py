import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="EduCompass | AI College Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Advanced CSS for Animations, Frosted Glass UI, and Gradients (Fixed unsafe_allow_html)
st.markdown("""
    <style>
    /* Gradient Background covering the full viewport */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #2e1065 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Header Card */
    .header-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2.5rem 1.5rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        animation: slideDown 1s ease-out;
    }
    
    .header-container h1 {
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-container p {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Input Card */
    .form-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 30px 50px rgba(0, 0, 0, 0.4);
        animation: fadeIn 1.2s ease-out;
    }
    
    /* Pulse Animated Prediction Output Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid #a78bfa;
        padding: 2.5rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 2.5rem;
        box-shadow: 0 0 30px rgba(167, 139, 250, 0.2);
        animation: pulseGlow 2.5s infinite alternate;
    }
    
    .pred-title {
        color: #38bdf8;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .pred-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.4;
    }

    /* Input Element Label Styling */
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Keyframes for CSS Animations */
    @keyframes slideDown {
        0% { opacity: 0; transform: translateY(-40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 20px rgba(167, 139, 250, 0.2); border-color: #a78bfa; }
        100% { box-shadow: 0 0 40px rgba(56, 189, 248, 0.4); border-color: #38bdf8; }
    }
    </style>
""", unsafe_allow_html=True)

# 3. App Header Widget (Fixed unsafe_allow_html)
st.markdown("""
    <div class="header-container">
        <h1>🎓 EduCompass AI</h1>
        <p>Intelligent College Prediction Engine powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# 4. Safely Load Pickled Model
@st.cache_resource
def load_college_model():
    try:
        with open("Modelclg.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("❌ 'Modelclg.pkl' file not found in the current directory. Please place your model file alongside app.py.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_college_model()

# 5. Form Wrapper & Layout Inputs (Arranged in a grid)
if model is not None:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.subheader("📊 Enter Academic Profile Details")
    st.caption("Note: All 9 features must be filled to fulfill model signature requirements.")
    
    # 3-Column Layout to neatly organize the 9 required fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feat1 = st.number_input("🎯 Entrance Percentile", min_value=0.0, max_value=100.0, value=92.5, step=0.01)
        feat4 = st.number_input("📝 12th Board (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
        feat7 = st.number_input("🏫 10th Board (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
        
    with col2:
        feat2 = st.number_input("🔢 Category Code (e.g. 1-5)", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        feat5 = st.number_input("📍 Region Code", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        feat8 = st.number_input("💰 Seat Type Preference", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
        
    with col3:
        feat3 = st.number_input("⚧ Gender (M, F)", min_value=0.0, max_value=1.0, value=0.0, step=1.0)
        feat6 = st.number_input("💼 Family Income Tier", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
        feat9 = st.number_input("⚙️ Stream Choice Code", min_value=0.0, max_value=20.0, value=3.0, step=1.0)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 6. Prediction Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    
    with btn_col:
        predict_clicked = st.button("✨ Predict Best Match Campus", use_container_width=True)
        
    # 7. Execution and Animation Rendering (Fixed unsafe_allow_html)
    if predict_clicked:
        # Group into a 1x9 matrix structure expected by Scikit-Learn
        features_array = np.array([[feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9]])
        
        with st.spinner("Analyzing model vectors & calculating nearest neighbors..."):
            try:
                prediction = model.predict(features_array)
                predicted_college = prediction[0]
                
                # Trigger confetti/balloons animation
                st.balloons()
                
                # Display output inside the styled glowing card
                st.markdown(f"""
                    <div class="prediction-card">
                        <div class="pred-title">🌟 Top Recommended Institution 🌟</div>
                        <div class="pred-value">{predicted_college}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Prediction Error: Ensure feature values fall within valid scaling ranges.")
                st.caption(f"Technical error context: {e}")
