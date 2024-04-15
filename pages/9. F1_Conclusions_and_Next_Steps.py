#!/usr/bin/env python
# coding: utf-8

# # 9. F1 Prediction Project - Conclusions and next steps

# # Table of Contents
# 1. [Overview](#overview)
# 2. [What Factors Most Influence the F1 Race Winner](#what-factors-most-influence-who-is-going-to-win-the-next-f1-race)
# 3. [Model Performance in Predicting F1 Winners](#how-did-the-model-perform-in-predicting-f1-winners)
# 4. [Financial Analysis of Model Predictions](#could-the-model-help-us-make-money-betting-on-f1-or-help-the-f1-betting-industry)
# 5. [Personal Learnings from the Project](#personal-learnings-from-the-project)
# 6. [Next Steps](#next-steps)
# 

# #### **Overview**

# In the introduction to our data science project on predicting F1 race winners, we set out to merge the thrill of motorsport analytics with the precision of machine learning. Through the course of this project, captured across several comprehensive notebooks, we’ve ventured into a world where large datasets—encompassing historical race outcomes, driver and team performance statistics, and race day conditions—become the fuel for our predictive models. From the initial explorations of raw data to the nuanced interpretations of model outcomes, this project was an exhilarating lap around the analytical circuit of Formula 1, designed to deliver strategic insights to teams and bettors alike, and to bring the fans closer to the action. By leveraging diverse data sources, including F1's own repositories, we've unpacked the predictors of racing success and showcased the impact of data science on enhancing betting strategies, optimizing team decisions, and elevating the fan experience. This project is a testament to the learning journey, illustrating that with the right guidance and tools, even those new to Python can make significant strides in a short time.

# #### **What factors most influence who is going to win the next F1 race?**

# <img src="feature_importance.png" alt="Influential_Factors_F1" width="800"/>

# Based on the insights gleaned from our model interpretability analysis, the key factors that most influence the likelihood of winning the next F1 race have been discerned. Primarily, a driver's starting grid position stands out as the paramount predictor, where a higher starting position correlates strongly with increased chances of winning. Furthermore, the accumulated points of drivers and constructors throughout the season significantly affect predictions, indicating that consistent performance is a strong indicator of future success. Experience in F1, denoted by years and races with each team, also plays a crucial role, with seasoned drivers and well-synergized team relationships having a notable impact on race outcomes. Lastly, the legacy and resources of powerhouse teams like Mercedes and Ferrari emerge as influential, with their historical success and technological prowess being pivotal factors in the model's predictions. Through the model interpretability process, we've uncovered that while past performance, strategic grid positions, and team dynamics hold weight in our model's estimations, the unpredictable and thrilling nature of F1 racing means every race day offers a new opportunity for surprises and upsets.

# ### **How did the model perform in predicting F1 winners?**

# ![Random Forest SMOTE F1](Random_Forest_SMOTE_F1.png)

# In our pursuit to forecast F1 race winners, the Random Forest model using SMOTE (Synthetic Minority Over-sampling Technique) for balancing our dataset emerged as the standout performer. As illustrated in the chart, this model achieved the highest Class 1 F1 Score—a harmonic mean of precision and recall—indicating a commendable balance between correctly predicting the true winners (recall) and the accuracy of those predictions (precision). Specifically, it attained an F1 score of 0.51, which, alongside a consistent model accuracy of 0.95 and the best recall of 0.50 for Class 1, underscores its effectiveness. The macro F1 score also peaks for this model, further affirming its superior predictive capability across both classes. Such results reinforce the Random Forest SMOTE model's robustness and reliability in identifying the potential victors in the thrilling races of F1, making it a valuable tool in the analytics arsenal of motorsports predictions.
# 
# The data science model performed impressively well for the year 2024 test set. The predictions made by the model for various races indicate a substantial understanding of the dynamics at play in the sport. For instance, our model accurately predicted the winner in the Saudi Arabian race with a confidence of 59%, and the Australian race with a 56% confidence level (e.g. hard predictions), leading to winnings from those events. In the other two races for 2024, it didn't give hard predictions, but the two drivers with the highest soft predictions, ended up winning the race. Again, this shows strong promise as tool to help work out who is likely to win the next F1 race. 
# 
# Despite the inherent unpredictability in sports like Formula 1, where outcomes can be influenced by a multitude of variables ranging from weather to mechanical reliability, our model has shown resilience and accuracy. The winnings column reveals that when the model was confident enough to suggest a bet, it resulted in positive returns. The performance of the model not only confirms the strength of the predictive algorithms but also the robustness of the underlying data and the efficiency of the processing.

# #### **Could the model help us make money betting on F1 or help the F1 Betting Industry?**

# ![would We Have Won Money](would_we_have_won_money.png)

# Analyzing the performance financially, the total earnings from the bets placed based on the model's predictions amounted to £92. Considering the initial bets were £200 (£100 each for the Saudi Arabian and Australian races), this represents a 46% return on investment. Such a return is notable, especially when no losses were recorded, which underscores the model's effectiveness in this context. The smart betting strategy employed here—betting only when the model's confidence was high—paid off, demonstrating the model’s potential as a decision-support tool in predictive sports analytics.
# 
# While our model has shown its potential in successfully predicting F1 race winners, it's important to stress that there are no guarantees in gambling—every bet carries a risk, regardless of the tools or data at hand. Although we saw positive results and a 46% return on investment with smart, model-informed betting strategies during this exercise, these outcomes are a best-case scenario and not a promise of consistent earnings. Betting, in any form, should be approached with caution, as outcomes can never be predicted with absolute certainty. Our model however can be a valuable aid, adding an edge to informed decision-making, but it should be used responsibly as part of a broader strategy that recognizes the inherent uncertainties of sports betting.

# #### **Personal Learnings from the Project**

# 1) **The amazing things you can do with data science** : In this data science project, it's been amazing to see how changing features and running different models, can change/improve the predictive outcomes (in my case with F1 races). I was really impressed/happy to see that, if I had followed my models output with the 2024 test set (& adhered to its 'hard predictions'), I could have financially benefited us throughout the 2024 racing season. This accomplishment to me underscores the tangible potential of applied machine learning in the sports betting/prediction arena. Overall, I'm just really excited to get more stuck into the field and see what other problems, I can bring my new data science skills to.
# 
# 2) **Using Data Science to explore a subject** : Embarking on this data science endeavor to predict F1 race winners has been a revelation, showcasing the profound capacity of data science to deepen our understanding of complex and dynamic subjects. Through the meticulous dissection of data, the application of machine learning models, and the interpretation of their outputs, I've gained insights into the nuances of Formula 1 that go far beyond surface-level knowledge. Data science has not only served as a powerful tool for prediction but has also emerged as an exceptional teacher, uncovering nuanced layers of strategy, skill, and chance that define the sport. This journey through data has transformed my perspective on F1, elevating my appreciation for the sport and its intricacies, and affirming data science's role as a key to unlocking the secrets held within any domain of passion or curiosity.
# 
# 3) **The challenges of data collection and the importance of documenting your processes** : The project also brought to light the intricate and laborious nature of data collection in some data science projects. Moving forward, I am eager to explore automated data collection methods to streamline this process (for this project), reducing the manual workload and mitigating the risks of human error inherent in data gathering. As I was new to Python and comfortable with Excel, I did a lot of work to manipulate date from different sources in Excel. Moving forward, I would like to keep improving my Python skills to be able to manipulate the data in notebooks, to make it easier for readers to follow (and recreate) my data gathering processes.
# 
# 4) **How helpful GenAI tools have been**: Entering the field as a complete beginner in Python, I'm in awe of the possibilities that unfold with supportive tools like ChatGPT and Google Gemini (& the guidance of the great educators from BrainStation). These resources, while not a replacement for the foundational knowledge of Python, have accelerated my learning journey, enabling me to make this project in a relatively short period. Without that support, I think it would have been very hard to deliver this project in the timeframe I had. One of the great rewards from the project, is the inspiration to now continue honing my coding skills (with and without GenAI), embracing the challenges and opportunities that lie ahead in the evolving landscape of data science.

# ### **Next Steps**

# Moving forward with the data science project on predicting Formula 1 race winners, I've outlined a roadmap that focuses on refinement and enhancement of our predictive capabilities. 
# 
# 1. **Make the model simpler** Complexity doesn't always equate to better performance, especially in real-time scenarios. We will focus on streamlining the model to ensure that it is not only accurate but also efficient, balancing between the depth of the data and the speed of prediction. Also the model currently has 181 columns, with many one-hot encoded variables. From our feature extraction from our Random Forest SMOTE model, we know what features were the most important to the model in terms of predictive power, so I would like to do a v2 of the model that is simpler (and therefore requires less compute).
# 
# 2. **Create other ML functions to predict pre-race variables** Alongside making it simpler, I would like to create some functions to help predict certain variables that we won't have before a race (e.g. if a driver is likely to crash, or a team is likely to have a mechanical issue), which significantly influence race outcomes. These enhancements are about moving our model to a more proactive, predictive model that can anticipate outcomes before the race begins.
# 
# 3. **Automate Data Collection Pipeline** is crucial for the model to be up-to-date with the latest information. When building the 2024 dataset this required a lot of manual work pulling data from different sites (e.g. F1.com, Wikipedia, & formula1points.com). By automating this process, we ensure that the data fed into the model is fresh and reflective of current dynamics, such as weather conditions, track details, and driver performance.
# 
# 4. **Improve UX for a customer facing application and intergrate with AWS** Finally from my side I would like to create a better user experience for the model, using AWS Amplify. AWS Amplify is a set of tools and services from Amazon Web Services that enables developers to build and deploy full-stack mobile and web applications that are scalable, secure, and integrate with AWS cloud services. AWS Amplify will help me set up a secure and scalable environment for the model and allow me to work on further integrating the data science skills I learned from BrainStation into a production environment. It will also facilitate better user interactions with the model's predictions, making the insights more accessible to stakeholders.
# 
# In summary, these next steps are designed to polish our model into a tool that's not only scientifically rigorous but also user-friendly and directly applicable to the dynamic world of Formula 1 racing.
