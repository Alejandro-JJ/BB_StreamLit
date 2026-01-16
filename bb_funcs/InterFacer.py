"""
Markdown customizer
"""
import streamlit as st
import matplotlib.colors as mcolors
xkcd = mcolors.XKCD_COLORS

# user defined colors


def InterFacer(colors, SLIDER_WIDTH):
    st.markdown(
        f"""
        <style>

        /* GLOBAL BACKGROUND */
        .stApp {{
            background-color: {colors["background"]};
        }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] {{
            background-color: {colors["sidebar"]};
        }}

        /* BUTTONS */
        div.stButton > button {{
            background-color: {colors["button_bg"]};
            color: {colors["button_text"]};
            border-radius: 6px;
            border: none;
        }}
        div.stButton > button:hover {{
            opacity: 0.85;
        }}

        /* SLIDER WIDTH — DO NOT TOUCH */
        div[data-testid="stSlider"] > div {{
            width: {SLIDER_WIDTH}px !important;
        }}

        /* SLIDER COLORS */
        div[data-baseweb="slider"] > div > div {{
            background-color: {colors["slider_track"]};
        }}
        div[data-baseweb="slider"] span {{
            background-color: {colors["slider_thumb"]};
        }}

        /* INPUT FIELDS */
        input, textarea {{
            background-color: {colors["input_bg"]} !important;
            color: {colors["input_text"]} !important;
        }}

        /* LABELS / TEXT */
        label, .stMarkdown {{
            color: {colors["label_text"]};
        }}

        </style>
        """,
        unsafe_allow_html=True
    )
