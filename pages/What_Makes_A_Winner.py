# project_conclusions.py
import streamlit as st

def app():
    st.title("F1 Prediction Project - What Makes A Winner")

    st.header("Overview")
    st.write("""
    Embarking on a quest within the high-octane world of Formula 1, this data science project's exploratory phase seeks to decode the myriad factors influencing race outcomes. Through a rigorous exploratory data analysis, we probe deep into a dataset brimming with the sport's storied legacy, tackling questions such as the impact of a team's prestige on victory likelihood, and whether historical success translates into future performance. 
    """)

    st.header("Which teams are the best to drive For?")
    st.image("Wins_by_team.jpg", caption="Wins by team")

    st.write("""
    The bar chart of Formula 1 team wins reveals pivotal insights for predictive analysis: top teams like Ferrari, Mercedes, and Red Bull have historically outperformed others in our dataset. While past success might help us understand potential future performance (e.g. top teams might be able to recruit the best drivers & engineers, have more funding, etc), it will be crucial to assess the statistical significance of team success as a predictive feature. 
    """)

    st.header("Which is the most successful engine manufacturer?")
    st.image("Wins_by_team.jpg", caption="Wins by position")

    st.write("""
    This chart offers a fascinating look into the dominance of engine manufacturers in Formula 1 over an unspecified period, crucial for understanding trends that could influence the outcome of races. Mercedes stands out with the highest number of wins, indicating their engines might give teams an edge. Ferrari and Renault follow, suggesting they are also competitive, while the presence of, Honda, Red Bull, BMW, Mugen-Honda, and Ford illustrates a more diverse field of engine suppliers with victories. Analyzing these patterns helps us predict which manufacturers could contribute to a team's success, as a superior engine often translates into a critical advantage on the track. This information is invaluable for our predictive modeling, as it underscores the significant role of engine performance in racing triumphs.
    """)

    st.header("How important is pole position?")
    st.image("Wins_By_position.jpg", caption="Wins by position")

    st.write("""
    This chart compellingly illustrates the impact of starting grid positions on winning rates in Formula 1 races. It shows a stark decline in win rates as the starting position moves away from the pole, with over 40% of wins coming from the pole position (1st place) itself. This sharply contrasts with the subsequent positions, which all have significantly lower win rates, emphasizing the pole position's advantage. The data clearly suggests that starting at the front of the grid significantly increases a driver's chances of winning, likely due to the clear track ahead allowing for an uncontested race pace. Understanding these trends is key in our predictive models, as the starting grid position could be a substantial predictor of race outcomes.
    """)

    st.header("Which Grand Prix is it most important to start from the front?")
    st.image("Top_Grand_Prix_by_Front.jpg", caption="Top_Grand_Prix_by_Front")

    st.write("""
    This chart presents a clear visualization of how pole position influences race outcomes across different Grand Prix. It's evident that the Spanish, Italian and Japanese Grand Prix see a higher percentage of wins from the pole position, suggesting that the track layout or other factors in these locations might give the leading starter a more pronounced advantage. Conversely, races like the British and Belgian Grand Prix show a lower reliance on pole position for a win, indicating that these tracks may allow for more overtaking or that strategy and car performance play a more significant role. This analysis is invaluable as it directs our modeling efforts towards considering the unique characteristics of each circuit and their impact on race strategy and outcomes.
    """)

    st.header("Does racing at home make a difference?")
    st.image("Home_race_wins.jpg", caption="Home_race_wins")

    st.write("""
    This chart illustrates the comparison between the home and non-home win rates for the top 10 Formula 1 drivers in the dataset. The blue bars represent each driver's win rate for races held in their home country, while the red bars show their win rate for races held outside their home country. This analysis can provide insights into whether being on home turf provides any advantage to drivers, which could be a significant factor in predicting race outcomes. The differences in win rates across drivers can also suggest the level of comfort and performance consistency in various racing environments. For example, it looks like Lewis Hamilton is very comfortable and performs very well at his home Grand Prix, whilst someone like Sergio Perez has not performed as well at his home race (maybe due to the large pressure he feels to perform in front of his home fans).
    """)

    st.header("Conclusion")
    st.write("""In conclusion, this data science project employs advanced analytics to predict Formula 1 race winners, taking into account a multitude of variables including driver performance, team strategy, car specifications, and track conditions. Through meticulous data processing and sophisticated modeling, we've gained valuable insights that not only enhance the understanding of the sport's dynamics but also give us a competitive edge in forecasting outcomes in the thrilling world of F1 racing.
    """)

# Note: This app function will be called from the main app managing navigation.
