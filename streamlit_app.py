import streamlit as st
import pandas as pd

# Load the data (replace with the path to your CSV file)
@st.cache
def load_data():
    data = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv')
    return data

df = load_data()

# Select a unique list of races to choose from
races = df['race'].unique()
selected_race = st.selectbox('Select a Race', races)

# Filter the dataframe for the selected race and calculate the top 3 drivers
def get_top_3_drivers(selected_race):
    race_df = df[df['race'] == selected_race]
    top_3_drivers = race_df.nlargest(3, 'prediction_probability')[['Driver', 'prediction_probability']]
    return top_3_drivers

if st.button('Show Top 3 Drivers'):
    top_3_drivers = get_top_3_drivers(selected_race)
    st.write(top_3_drivers)

# To run the Streamlit app, save this code in a file app.py and run it with:
# streamlit run app.py
