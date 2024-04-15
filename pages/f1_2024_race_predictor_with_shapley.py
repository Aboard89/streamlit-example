import streamlit as st
import pandas as pd
import shap

st.title('2024 F1 Race Predictions')
st.write("""
         Welcome to the 2024 F1 Race Predictions app. Here you can select a specific race to see 
         the top 3 predicted drivers based on our model's prediction probability. Simply choose a race 
         from the dropdown below and click "Show Top 3 Drivers" to view the predictions and see the 
         corresponding SHAP force plot displaying their impact on the prediction.
         """)

# Load the race predictions data
@st.cache
def load_data():
    data = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
    return data

df = load_data()

# Load the Shapley values data
@st.cache
def load_shap_values():
    shap_values = pd.read_csv('shap_values.csv')
    return shap_values

shap_df = load_shap_values()

# Select a unique list of races to choose from
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

# Filter the dataframe for the selected race and calculate the top 3 drivers
def get_top_3_drivers(selected_race):
    race_df = df[df['race'] == selected_race]
    top_3_drivers = race_df.nlargest(3, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_3_drivers

if st.button('Show Top 3 Drivers'):
    top_3_drivers = get_top_3_drivers(selected_race)
    st.write(top_3_drivers)
    
    # Find the top driver and corresponding index
    top_driver = top_3_drivers.iloc[0]
    driver_index = top_driver['index']
    
    # Filter Shapley values for the top driver using the index
    driver_shap_values = shap_df.loc[driver_index]
    
    # Assuming the Shapley values and the features data is loaded correctly
    # Initiate Javascript for visualization
    shap.initjs()
    
    # Create a force plot for the top driver's SHAP values
    # You may need to adjust indices and structures depending on your actual data setup
    force_plot = shap.force_plot(
        explainer.expected_value[1],  # Use the expected value for the positive class
        driver_shap_values.values,    # SHAP values for the top driver (ensure this matches your data structure)
        feature_names=shap_df.columns # Assuming feature names are in the SHAP DataFrame
    )
    
    # Display the SHAP force plot in Streamlit
    st_shap(force_plot, height=300)

# Helper function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# To run the Streamlit app, save this code in a file app.py and run it with:
# streamlit run app.py
