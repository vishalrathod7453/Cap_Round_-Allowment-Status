import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EduCompass AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("Modelclg.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

h1, h2, h3, h4 {
    color: white;
    text-align: center;
}

.big-title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
    animation: glow 2s infinite alternate;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
    animation: fadeIn 1s ease-in;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 20px;
    border: none;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}

.result-box {
    background-color: #16a34a;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
    animation: fadeIn 1s ease-in;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #38bdf8;
    }
    to {
        text-shadow: 0 0 20px #0ea5e9;
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🎓 EduCompass AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Intelligent College Prediction Engine powered by Machine Learning</div>',
    unsafe_allow_html=True
)

# ---------------- FORM CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Enter Academic Profile Details")

col1, col2 = st.columns(2)

with col1:

    feat1 = st.slider(
        "🎯 Entrance Percentile",
        0.0, 100.0, 92.5
    )

    feat3 = st.slider(
        "📝 12th Board Percentage",
        0.0, 100.0, 85.0
    )

    feat4 = st.slider(
        "🏫 10th Board Percentage",
        0.0, 100.0, 90.0
    )

    # CATEGORY
    category = st.selectbox(
        "📂 Select Category",
        ["OPEN", "OBC", "VJNT", "SC", "ST", "NT", "SBC"]
    )

    category_mapping = {
        "OPEN": 0,
        "OBC": 1,
        "VJNT": 2,
        "SC": 3,
        "ST": 4,
        "NT": 5,
        "SBC": 6
    }

    feat2 = category_mapping[category]

    # REGION
    region = st.selectbox(
        "📍 Select Region",
        ["Urban", "Rural", "Semi-Urban"]
    )

    region_mapping = {
        "Urban": 0,
        "Rural": 1,
        "Semi-Urban": 2
    }

    feat5 = region_mapping[region]

with col2:

    # SEAT TYPE
    seat = st.selectbox(
        "💺 Seat Type Preference",
        ["Government", "Private", "Management"]
    )

    seat_mapping = {
        "Government": 0,
        "Private": 1,
        "Management": 2
    }

    feat6 = seat_mapping[seat]

    # GENDER
    gender = st.radio(
        "⚧ Gender",
        ["Male", "Female", "Other"]
    )

    gender_mapping = {
        "Male": 0,
        "Female": 1,
        "Other": 2
    }

    feat7 = gender_mapping[gender]

    # FAMILY INCOME
    income = st.selectbox(
        "💼 Family Income Tier",
        [
            "Below 1 Lakh",
            "1 - 5 Lakh",
            "5 - 10 Lakh",
            "Above 10 Lakh"
        ]
    )

    income_mapping = {
        "Below 1 Lakh": 0,
        "1 - 5 Lakh": 1,
        "5 - 10 Lakh": 2,
        "Above 10 Lakh": 3
    }

    feat8 = income_mapping[income]

    # STREAM
    stream = st.selectbox(
        "⚙️ Stream Choice",
        [
            "Computer Engineering",
            "IT",
            "Mechanical",
            "Civil",
            "AI & DS",
            "Electronics"
        ]
    )

    stream_mapping = {
        "Computer Engineering": 0,
        "IT": 1,
        "Mechanical": 2,
        "Civil": 3,
        "AI & DS": 4,
        "Electronics": 5
    }

    feat9 = stream_mapping[stream]

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION BUTTON ----------------
if st.button("🚀 Predict Best College"):

    input_data = np.array([[
        feat1,
        feat2,
        feat3,
        feat4,
        feat5,
        feat6,
        feat7,
        feat8,
        feat9
    ]])

    with st.spinner("🔍 Analyzing your academic profile..."):
        time.sleep(2)

        prediction = model.predict(input_data)

    st.balloons()

    st.markdown(
        f"""
        <div class="result-box">
            🎯 Predicted College: {prediction[0]}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
    <h4 style='color:#94a3b8'>
    Built with ❤️ using Streamlit & Machine Learning
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
