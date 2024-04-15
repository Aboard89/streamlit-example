import streamlit as st
import pandas as pd
import pickle

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'random_forest_grid_search.pkl'
    try:
        with open(model_path, 'rb') as file:
            grid_search_cv = pickle.load(file)
        return grid_search_cv.best_estimator_
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

best_pipeline = load_model()

if best_pipeline is None:
    st.stop()

st.title('2024 F1 Race Predictions')

# Load race data
df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')

# Dropdown and race selection
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

def get_top_driver(selected_race):
    race_df = df[df['race'] == selected_race]
    top_driver = race_df.nlargest(1, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_driver.iloc[0]

if st.button('Show Top Driver'):
    top_driver = get_top_driver(selected_race)
    st.write('Top Driver:', top_driver)
