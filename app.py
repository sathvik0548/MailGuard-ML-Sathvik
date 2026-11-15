
"""MailGuard ML - Streamlit entry point.

Run locally: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MailGuard ML | Sathvik",
    page_icon=":material/shield:",
    layout="wide"
)

from src.app_resources import get_model_bundle, get_service

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #050505;
    color: #f5f5f5;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #252525;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] {
    color: #999999 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid #292929;
    border-radius: 12px;
    padding: 16px;
}

/* Buttons */
.stButton > button {
    background-color: #151515;
    color: #ffffff;
    border: 1px solid #333333;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
}

/* Button hover */
.stButton > button:hover {
    border-color: #3b82f6;
    color: #ffffff;
}

/* Text areas */
.stTextArea textarea {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    border-radius: 10px !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #151515 !important;
    border-color: #333333 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #bbbbbb !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #111111;
    border: 1px solid #292929;
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #292929;
    border-radius: 8px;
}

/* Horizontal dividers */
hr {
    border-color: #292929 !important;
}

/* Links */
a {
    color: #60a5fa !important;
}

/* Sidebar navigation */
[data-testid="stSidebarNav"] {
    background-color: #0a0a0a;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #202020;
    border-radius: 7px;
    border-left: 3px solid #3b82f6;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #050505;
}

::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555555;
}

</style>
""", unsafe_allow_html=True)

pages = [
    st.Page(
        "views/analyzer.py",
        title="Email Analyzer",
        icon=":material/mail:",
        default=True
    ),
    st.Page(
        "views/evaluation.py",
        title="Model Evaluation",
        icon=":material/analytics:"
    ),
    st.Page(
        "views/monitoring.py",
        title="Monitoring",
        icon=":material/monitor_heart:"
    ),
    st.Page(
        "views/model_info.py",
        title="Model Information",
        icon=":material/info:"
    ),
]

get_service()
model, model_error = get_model_bundle()

with st.sidebar:
    st.markdown("### :material/shield: MailGuard ML")

    if model is not None:
        st.success(
            f"Model ready: {model.display_name} v{model.version}",
            icon=":material/check_circle:"
        )
    else:
        st.error(
            "Model unavailable - see Model Information.",
            icon=":material/error:"
        )

    st.caption(
        "Educational spam-detection demo trained on the public SpamAssassin corpus. "
        "Emails you analyse are processed in memory and are not stored or logged."
    )

st.navigation(pages).run()