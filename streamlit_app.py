import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

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

# Display race winner
st.write("Race Winner:", race_winner)

# Get predicted race winner
predicted_winner_data = filtered_data[filtered_data['Predicted_Winner'] == 1]
if not predicted_winner_data.empty:
    predicted_winner = predicted_winner_data['driver_name'].iloc[0]
else:
    predicted_winner = 'No predicted winner data'

# Display predicted race winner
st.write("Predicted Race Winner:", predicted_winner)

# Embedding Wikipedia.org using an iframe
# Note: Wikipedia might not load due to CSP restrictions
iframe_code = f"""
<iframe src="https://www.wikipedia.org/" width="100%" height="400">
  Your browser does not support iframes.
</iframe>
"""
components.html(iframe_code, height=400)
