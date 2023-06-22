import streamlit.components.v1 as components
import streamlit as st
# ======================================================================
# Change page title and page icon
st.set_page_config(
    page_title="TodAI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# hide "made by streamlit"
hide_menu_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# ======================================================================
