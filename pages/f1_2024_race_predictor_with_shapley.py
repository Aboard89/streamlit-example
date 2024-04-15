# Load the race predictions data
@st.cache
def load_data():
    data = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
    data.reset_index(inplace=True, drop=False)  # Reset the index and add it as a column
    return data

df = load_data()

# Function to get the top driver for the selected race
def get_top_driver(selected_race):
    race_df = df[df['race'] == selected_race]
    top_driver = race_df.nlargest(1, 'prediction_probability')[['Driver', 'prediction_probability', 'index']]
    return top_driver

if st.button('Show Top Driver'):
    top_driver = get_top_driver(selected_race)
    st.write(top_driver)
    
    # Extracting the index from the top driver row
    driver_index = top_driver['index'].values[0]  # Ensure this now refers to the actual column 'index'
    
    # Filter Shapley values for the top driver using the index
    driver_shap_values = shap_df.loc[driver_index]
    
    # Initiate Javascript for visualization
    shap.initjs()
    
    # Create and display the force plot for the top driver's SHAP values
    force_plot = shap.force_plot(
        explainer.expected_value[1],  # Use the expected value for the positive class
        driver_shap_values.values,    # SHAP values for the top driver
        feature_names=shap_df.columns # Assuming feature names are in the SHAP DataFrame
    )
    
    # Helper function to display SHAP plots in Streamlit
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        st.components.v1.html(shap_html, height=height)

    st_shap(force_plot, height=300)
