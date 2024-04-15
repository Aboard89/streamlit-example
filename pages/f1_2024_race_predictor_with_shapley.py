import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title('2024 F1 Race Predictions')
st.write("""
         Welcome to the 2024 F1 Race Predictions app. Here you can select a specific race to see 
         the top predicted driver based on our model's prediction probability. Simply choose a race 
         from the dropdown below and click "Show Top Driver" to view the predictions and see the 
         corresponding SHAP force plot displaying their impact on the prediction.
         """)

# Load data functions
@st.cache
def load_data():
    data = pd.read_csv('/mnt/data/2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
    return data

@st.cache
def load_shap_values():
    shap_values = pd.read_csv('/mnt/data/shap_values.csv', encoding='ISO-8859-1')
    return shap_values

df = load_data()
shap_df = load_shap_values()

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

    # Extracting the index from the top driver row
    driver_index = top_driver['index']
    
    # Filter Shapley values for the top driver using the index
    driver_shap_values = shap_df.iloc[driver_index]

    # Create a force plot for the top driver's SHAP values
    explainer = shap.Explainer(lambda x: x)  # Dummy explainer for demonstration
    shap_values = explainer.shap_values(driver_shap_values.iloc[:-1])  # Assume last column is output
    force_plot = shap.force_plot(
        explainer.expected_value,  # Use the expected value for the positive class
        shap_values,  # SHAP values for the top driver
        feature_names=driver_shap_values.index[:-1]  # Using index as feature names, omit last column if it's output
    )

    # Convert SHAP plot to HTML and display in Streamlit
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=300)
