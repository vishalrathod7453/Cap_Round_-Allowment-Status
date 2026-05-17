import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #141e30, #243b55);
    }

    .stButton>button {
        background-color: #00c6ff;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #0072ff;
        transform: scale(1.03);
    }

    .title {
        text-align: center;
        font-size: 45px;
        color: white;
        font-weight: bold;
    }

    .subtext {
        text-align: center;
        color: #dcdcdc;
        font-size: 18px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(255,255,255,0.1);
        color: white;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# LOAD LOTTIE ANIMATION
# ------------------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_xRmNN8.json"
)

    )
