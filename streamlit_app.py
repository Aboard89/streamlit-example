# app.py or streamlit_app.py
import streamlit as st
pip install matplotlib
import matplotlib.pyplot as plt
from pages import race_winner, top_3_drivers  # Assumes 'race_winner.py' and 'top_3_drivers.py' are inside a 'pages' folder

PAGES = {
    "Race Winner": race_winner,
    "Top 3 Drivers": top_3_drivers
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
