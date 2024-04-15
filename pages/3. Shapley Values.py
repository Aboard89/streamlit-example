import streamlit as st
import pandas as pd
import shap
import pickle

def load_model():
    model_path = 'random_forest_grid_search.pkl'
    with open(model_path, 'rb') as file:
        grid_search_cv = pickle.load(file)
    return grid_search_cv.best_estimator_

best_pipeline = load_model()
rf_model = best_pipeline.named_steps['clf']
scaler = best_pipeline.named_steps['scl']

shap_values_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')
if 'output' in shap_values_df.columns:
    shap_values_df = shap_values_df.drop(columns=['output'])
feature_names = shap_values_df.columns

st.title('F1 Race Prediction SHAP Value Plot')

index_number = st.number_input('Enter the index number from the F1 Race prediction app:', min_value=0, format='%d')

if st.button('Generate SHAP Plot'):
    if index_number >= len(shap_values_df):
        st.error(f"Index is out of bounds. Please enter a number between 0 and {len(shap_values_df)-1}.")
    else:
        try:
            driver_shap_values = shap_values_df.iloc[index_number].values.reshape(1, -1)
            driver_features_transformed = scaler.transform(driver_shap_values)
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(driver_features_transformed)

            force_plot = shap.force_plot(
                explainer.expected_value[1], 
                shap_values[1],
                driver_features_transformed,
                feature_names=feature_names
            )

            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            st.components.v1.html(shap_html, height=300)

        except Exception as e:
            st.error(f"An error occurred: {e}")
