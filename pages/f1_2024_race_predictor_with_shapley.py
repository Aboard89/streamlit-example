import streamlit as st
import pandas as pd
import shap
import pickle
import os

# Load the complete GridSearchCV model from a .pkl file
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'random_forest_grid_search.pkl'
    try:
        with open(model_path, 'rb') as file:
            grid_search_cv = pickle.load(file)
        return grid_search_cv.best_estimator_  # Access the best estimator of the GridSearch
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

best_pipeline = load_model()

if best_pipeline is None:
    st.stop()

st.title('2024 F1 Race Predictions')

# Load data
df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
shap_df = pd.read_csv('shap_values.csv', encoding='ISO-8859-1')

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
    if 'output' in shap_df.columns:
        driver_features = shap_df.loc[[driver_index]].drop('output', axis=1)
    else:
        st.error("'output' column not found in SHAP DataFrame.")
        st.stop()

    # Assuming default naming, access the RandomForestClassifier with 'randomforestclassifier'
    rf_model = best_pipeline.named_steps['randomforestclassifier']
    explainer = shap.TreeExplainer(rf_model)

    # Since 'driver_features' need to be 2D, ensure they are in the correct shape
    if len(driver_features.shape) == 1:
        driver_features = driver_features.values.reshape(1, -1)
    shap_values = explainer.shap_values(driver_features)

    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], driver_features, link='logit')
    shap_html = f"<head>{shap.getjs()}</head><body>{shap.force_plot(explainer.expected_value[1], shap_values[1], driver_features, link='logit', matplotlib=True).data}</body>"
    st.components.v1.html(shap_html, height=300)
