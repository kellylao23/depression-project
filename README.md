# Early Detection of Major Depressive Disorder (MDD) in Adolescents

## Overview
Major depressive disorder (MDD) is a common disorder among adolescents, with lifetime recurrence rates of ~70%. MDD has been on the rise in recent years, impacting adolescents particularly with a 63% increase from 2013 to 2016. However, early signs of depression often go unnoticed among adolescents; moreover fears of stigmatization and concerns of confidentiality impose barriers for them to seek help. 

Thus this project aims to provide a quick and less intimidating approach to early detection of MDD with the following hypotheses: 
1. MDD can be predicted by variables from Electronic Health Records and demographics/lifestyle.
2. Machine learning (ML) models offer a great approach to analyze the relationship between depression and such variables.
3. ML models produces meaningful insights that enhance the understanding of driving factors for MDD and facilitate appropriate early intervention.
4. Anxiety is a close and less stigmatized proxy, which can be utilized to predict depression.

# Dataset
The data contained 62 attributes of 4184 students from Nice (France) University. In Jupyter Notebook, Python script was written for EDA, feature engineering and data preprocessing. Feature selection,  model training,  and hyperparameter tuning were done on 75% of the data while the rest 25% were used for model validation and evaluation. Models were trained using top 12 features and their performance were compared using metrics like accuracy, AUC, etc. The chosen model (XGBoost) was implemented via a web app developed using Streamlit which produces predicted risk of MDD along with its driving factors. The analysis concluded all four hypotheses were supported.

# Web Application
An interactive Streamlit app `streamlit_app.py` uses the XGBoost model trained on the dataset to predict the likelihood of MDD in adolescents. It uses the data that the user inputs -- demographic and electronic health record (EHR) data and displays a comprehensive summary of factors that are highly correlated and less correlated with MDD. 

The application features:
- **Home page** explaining the context and motivation of the project
- **Exploratory Data Analysis (EDA) page** for exploration of the dataset used to train the model
- **Prediction page** where users input their own data and receive a personalized prediction with SHAP-based model explainability
