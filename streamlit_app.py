# app.py or streamlit_app.py
import streamlit as st
from pages import f1_2024_race_predictor, f1_2024_top_driver, Shapley_Values, What_Makes_A_Winner, project_conclusions  # Assumes 'race_winner.py' and 'top_3_drivers.py' are inside a 'pages' folder

PAGES = {
    "Top 3 Drivers": f1_2024_race_predictor,
    "Top Driver": f1_2024_top_driver,
    "Shapley Values": Shapley_Values,
    "What makes a winner?": What_Makes_A_Winner,
    "Project Conclusion": project_conclusions,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
