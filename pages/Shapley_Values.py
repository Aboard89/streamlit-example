# Shapley_Values.py
import streamlit as st
import pandas as pd
import shap
import pickle

def load_model():
    model_path = 'random_forest_grid_search.pkl'
    try:
        with open(model_path, 'rb') as file:
            grid_search_cv = pickle.load(file)
        return grid_search_cv.best_estimator_
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

def generate_shap_plot(index_number):
    try:
        shap_values_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')
        if not (0 <= index_number < len(shap_values_df)):
            st.error(f"Index is out of bounds. Please enter a number between 0 and {len(shap_values_df)-1}.")
            return
        
        shap_values_row = shap_values_df.iloc[index_number]
        shap_values_to_plot = shap_values_row.values  # Assuming all columns are SHAP values
        
        # Set a baseline probability for the plot
        base_value = 0.501  # Adjust this as necessary

        force_plot = shap.force_plot(
            base_value,
            shap_values_to_plot,
            feature_names=shap_values_df.columns
        )

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.components.v1.html(shap_html, height=300)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

def app():
    st.title('F1 Race Prediction SHAP Value Plot')

    best_pipeline = load_model()
    if best_pipeline is None:
        return  # Stop execution if the model could not be loaded

    index_number = st.number_input('Enter the index number from the F1 Race prediction app:', min_value=0, value=0, format='%d')
    if st.button('Generate SHAP Plot'):
        generate_shap_plot(index_number)

# Note: This app function will be called from the main app managing navigation.
