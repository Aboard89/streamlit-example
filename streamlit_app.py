import streamlit as st
import pandas as pd

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('Streamlit_dataset.csv')
    return data

data = load_data()

# Sidebar - Year selection
selected_year = st.sidebar.selectbox('Select a Year', data['year'].unique())

# Sidebar - Location selection based on the year
filtered_data_by_year = data[data['year'] == selected_year]
selected_location = st.sidebar.selectbox('Select a Location', filtered_data_by_year['location'].unique())

# Filtering data based on selection
filtered_data = filtered_data_by_year[filtered_data_by_year['location'] == selected_location]

# Get race winner
race_winner = filtered_data[filtered_data['race_win'] == 1]['driver_name'].iloc[0]

# Get predicted race winner
predicted_winner = filtered_data[filtered_data['Predicted_Winner']].iloc[0]['driver_name']

# Display the results
st.write(f"Actual Race Winner for {selected_year} in {selected_location}: {race_winner}")
st.write(f"Predicted Race Winner for {selected_year} in {selected_location}: {predicted_winner}")
