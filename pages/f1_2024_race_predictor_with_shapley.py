import streamlit as st
import pandas as pd
import shap
import pickle
import os

# Load the model from a .pkl file
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'random_forest_grid_search.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title('2024 F1 Race Predictions')
st.write("""
         Welcome to the 2024 F1 Race Predictions app. Here you can select a specific race to see 
         the top predicted driver based on our model's prediction probability. Simply choose a race 
         from the dropdown below and click "Show Top Driver" to view the predictions and see the 
         corresponding SHAP force plot displaying their impact on the prediction.
         """)

# Load data
df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
shap_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')

# Dropdown to select race
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

# Function to get the top driver for the selected race
def get_top_driver(selected_race):
    race_df = df[df['race'] == selected_race]
    top_driver = race_df.nlargest(1, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_driver.iloc[0]

if st.button('Show Top Driver'):
    top_driver = get_top_driver(selected_race)
    st.write(top_driver)

    # Filter Shapley values for the top driver using the index from top_driver
    driver_index = top_driver['index']
    driver_features = shap_df.iloc[driver_index].drop('output', axis=1)  # assuming 'output' is the prediction column

    # Create a SHAP explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(driver_features)

    # Create a force plot for the top driver's SHAP values
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], driver_features)

    # Convert SHAP plot to HTML and display in Streamlit
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=300)
