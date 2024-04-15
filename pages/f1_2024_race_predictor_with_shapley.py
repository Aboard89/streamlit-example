import streamlit as st
import pandas as pd
import shap

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
    data = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
    data['index'] = data.index  # Ensure index is a column
    return data

df = load_data()

# Load the Shapley values data
@st.cache
def load_shap_values():
    shap_values = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')
    return shap_values

shap_df = load_shap_values()

# Dropdown to select race
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

# Function to get the top driver for the selected race
def get_top_driver(selected_race):
    race_df = df[df['race'] == selected_race]
    print(race_df.columns)  # Debugging to check column names
    top_driver = race_df.nlargest(1, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_driver.iloc[0]  # Return the top driver as a Series

if st.button('Show Top Driver'):
    top_driver = get_top_driver(selected_race)
    st.write(top_driver)

    # Extracting the index from the top driver row
    driver_index = top_driver['index']
    
    # Filter Shapley values for the top driver using the index
    driver_shap_values = shap_df.iloc[driver_index]

    # SHAP plotting
    explainer = shap.Explainer(lambda x: x)  # Dummy explainer
    shap_values = explainer.shap_values(driver_shap_values.iloc[:-1])  # Assume last column is output
    force_plot = shap.force_plot(explainer.expected_value, shap_values, feature_names=driver_shap_values.index[:-1])

    # Display SHAP plot
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=300)
