import streamlit as st
import pandas as pd

# Load the data
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
race_winner_data = filtered_data[filtered_data['race_win'] == 1]
if not race_winner_data.empty:
    race_winner = race_winner_data['driver_name'].iloc[0]
else:
    race_winner = 'No winner data'

# Get predicted race winner
predicted_winner_data = filtered_data[filtered_data['Predicted_Winner']]
if not predicted_winner_data.empty:
    predicted_winner = predicted_winner_data['driver_name'].iloc[0]
else:
    predicted_winner = 'No predicted winner data'

# Display the results
st.write(f"Actual Race Winner for {selected_year} in {selected_location}: {race_winner}")
st.write(f"Predicted Race Winner for {selected_year} in {selected_location}: {predicted_winner}")

