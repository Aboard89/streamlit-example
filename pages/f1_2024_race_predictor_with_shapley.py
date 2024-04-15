import streamlit as st
import pandas as pd
import shap
import pickle
import os

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

# Load race data and shap values
df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
shap_values_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')

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

    driver_index = top_driver['index'].astype(int)
    # Get the SHAP values for the top driver
    driver_shap_values = shap_values_df.iloc[driver_index, :-1]  # Excluding the last column if it's not a feature
    # Create a force plot for the top driver's SHAP values
    rf_model = best_pipeline.named_steps['randomforestclassifier']
    explainer = shap.TreeExplainer(rf_model)
    
    # Assuming that the last column of shap_values_df is the output value, which we do not need for the plot
    shap_values = explainer.shap_values(shap_values_df.drop(columns=['output']).iloc[driver_index])
    
    # If you need to transform features (e.g., scaling) before plotting, do it here.
    # driver_features_transformed = best_pipeline.named_steps['standardscaler'].transform(driver_features)
    
    # Now we plot, using the correct expected_value index if it's a classification problem
    force_plot = shap.force_plot(
        explainer.expected_value[1],  # Index [1] for the positive class; use [0] for the negative class if binary classification
        shap_values[1],  # Same here for the positive class
        feature_names=shap_values_df.drop(columns=['output']).columns  # Adjust if your DataFrame structure is different
    )
    
    # Convert the plot to HTML
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    
    # Display in Streamlit
    st.components.v1.html(shap_html, height=300)
