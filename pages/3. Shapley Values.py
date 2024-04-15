import streamlit as st
import pandas as pd
import shap
import pickle  # Make sure this import statement is included

st.title('F1 Race Prediction SHAP Value Plot')

# Function to load the model
def load_model():
    model_path = 'random_forest_grid_search.pkl'
    with open(model_path, 'rb') as file:
        grid_search_cv = pickle.load(file)  # Ensure 'pickle' is imported
    return grid_search_cv.best_estimator_

# Check if the 'load_model' function is being called correctly
try:
    best_pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to generate SHAP plot from precomputed SHAP values
def generate_shap_plot(index_number):
    try:
        # Load SHAP values from the csv
        shap_values_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')
        
        # Check if the index number is within the range of the DataFrame
        if not (0 <= index_number < len(shap_values_df)):
            st.error(f"Index is out of bounds. Please enter a number between 0 and {len(shap_values_df)-1}.")
            return

        # Select the row with the SHAP values for the chosen index
        shap_values_row = shap_values_df.iloc[index_number]

        # Assuming your CSV has columns for SHAP values only
        shap_values_to_plot = shap_values_row.values
        
        # Set the baseline as 0.05 for the probability
        base_value = 0.501

        # Generate the SHAP force plot
        force_plot = shap.force_plot(
            base_value,
            shap_values_to_plot,
            feature_names=shap_values_df.columns
        )
        
        # Convert the plot to HTML and display it
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.components.v1.html(shap_html, height=300)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit UI for input and button
index_number = st.number_input('Enter the index number from the F1 Race prediction app:', min_value=0, value=0, format='%d')
if st.button('Generate SHAP Plot'):
    generate_shap_plot(index_number)
