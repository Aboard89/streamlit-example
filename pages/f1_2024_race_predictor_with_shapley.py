import streamlit as st
import pandas as pd
import shap
import pickle
import os

# Load the complete pipeline model from a .pkl file
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'random_forest_grid_search.pkl'  # Ensure this is the correct path
    try:
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

pipeline = load_model()

if pipeline is None:
    st.stop()

st.title('2024 F1 Race Predictions')

# Load data
df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
shap_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')

# Log or display column names for troubleshooting
st.write("SHAP DataFrame columns:", shap_df.columns)

# Dropdown and race selection
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

def get_top_driver(selected_race):
    race_df = df[df['race'] == selected_race]
    top_driver = race_df.nlargest(1, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_driver.iloc[0]

if st.button('Show Top Driver'):
    top_driver = get_top_driver(selected_race)
    st.write(top_driver)

    driver_index = top_driver['index']
    # Check if 'output' column exists and then drop it
    if 'output' in shap_df.columns:
        driver_features = shap_df.loc[[driver_index]].drop('output', axis=1)
    else:
        st.error("'output' column not found in SHAP DataFrame.")
        driver_features = shap_df.loc[[driver_index]]  # Proceed without dropping

    # Assuming the pipeline includes necessary transformers and the final classifier
    rf_model = pipeline.named_steps['clf']  # Check your pipeline configuration
    explainer = shap.TreeExplainer(rf_model)

    # Generate SHAP values
    shap_values = explainer.shap_values(driver_features)

    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], driver_features, link='logit')
    shap_html = f"<head>{shap.getjs()}</head><body>{shap.force_plot(explainer.expected_value[1], shap_values[1], driver_features, link='logit', matplotlib=True).data}</body>"
    st.components.v1.html(shap_html, height=300)
