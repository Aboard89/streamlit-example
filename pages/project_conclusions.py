import streamlit as st
import os

st.title("F1 Prediction Project - Conclusions and Next Steps")

st.header("Overview")
st.write("""
In this data science project, we've used machine learning to predict F1 race winners, combining motorsport analytics with advanced data techniques. The project has taken us through rigorous data analysis, exploring numerous factors that influence race outcomes and providing insights that can aid betting strategies and team decisions.
""")

st.header("Key Insights on Influential Factors")
st.image("feature_importance.png", caption="Influential Factors in F1 Race Outcomes")

st.write("""
Starting grid position, accumulated points throughout the season, and team dynamics are crucial in predicting race winners. Our analysis highlights the significant impact of these factors on race outcomes.
""")

st.header("Model Performance")
st.image("best_model.png", caption="Model Performance in Predicting F1 Winners")

st.write("""
Our Random Forest model with SMOTE achieved impressive accuracy and F1 scores, making it a reliable tool for predicting race winners. The model's predictions for the 2024 season showed promising results, accurately forecasting winners with high confidence.
""")

st.header("Financial Analysis of Model Predictions")
st.image("betting_results.png", caption="Financial Outcomes Based on Model Predictions")

st.write("""
The model's predictions resulted in a 46% return on investment for selected races, demonstrating its potential to assist in profitable betting strategies. However, betting carries risks, and outcomes can never be guaranteed.
""")

st.header("Personal Learnings from the Project")
st.write("""
1. **Data Science Application:** The project demonstrated the powerful application of data science in sports analytics.
2. **Explorative Data Science:** It also highlighted how data science can be used to explore and understand complex subjects like Formula 1.
3. **Data Collection Challenges:** The importance of efficient data collection and process documentation was a key learning point.
""")

st.header("Next Steps")
st.write("""
- **Simplify the Model:** Reduce complexity to enhance real-time performance.
- **Predict Pre-race Variables:** Develop functions to predict key pre-race variables like crash likelihood.
- **Automate Data Collection:** Streamline data gathering from various sources to keep the model updated.
- **Improve User Experience:** Enhance the interface and integrate with AWS for better accessibility and scalability.
""")

st.write("This project not only refined our predictive capabilities but also set the stage for further enhancements that will make the tool more user-friendly and applicable in real-world scenarios.")
