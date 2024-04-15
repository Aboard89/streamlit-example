# app.py or streamlit_app.py
import streamlit as st
from pages import race_winner, top_3_drivers  # Assumes 'race_winner.py' and 'top_3_drivers.py' are inside a 'pages' folder

PAGES = {
    "Top 3 Drivers": 1. f1_2024_race_predictor,
    "Top Driver": 2. f1_2024_top_driver,
    "Shapley Values": 3. Shapley Values,
    "Project Conclusion": 4. project_conclusions
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
