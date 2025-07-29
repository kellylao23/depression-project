# Early Detection of Major Depressive Disorder (MDD) in Adolescents

## Overview
Major depressive disorder (MDD) is a common disorder among adolescents, with lifetime recurrence rates of ~70%. MDD has been on the rise in recent years, impacting adolescents particularly with a 63% increase from 2013 to 2016. However, early signs of depression often go unnoticed among adolescents; moreover fears of stigmatization and concerns of confidentiality impose barriers for them to seek help. 

Thus this project aims to provide a quick and less intimidating approach to early detection of MDD with the following hypotheses: 
1. MDD can be predicted by variables from Electronic Health Records and demographics/lifestyle.
2. Machine learning (ML) models offer a great approach to analyze the relationship between depression and such variables.
3. ML models produces meaningful insights that enhance the understanding of driving factors for MDD and facilitate appropriate early intervention.
4. Anxiety is a close and less stigmatized proxy, which can be utilized to predict depression.

# Dataset
The data are Electronic Health Records (EHRs) obtained during a compulsory medical visit of undergraduate students (primarily freshmen and sophomores) at the university medical service in Nice, France. This was collected in the CALCIUM database (Consultations Assistés par Logiciel pour les Centres Inter-Universitaire de Médecine) and included basic information about the students as well as the target variable of depressive symptoms. 

The data contained 62 attributes of 4184 students from Nice (France) University. In Jupyter Notebook, Python script was written for EDA, feature engineering and data preprocessing. Feature selection,  model training,  and hyperparameter tuning were done on 75% of the data while the rest 25% were used for model validation and evaluation. Models were trained using top 12 features and their performance were compared using metrics like accuracy, AUC, etc. The chosen model (XGBoost) was implemented via a web app developed using Streamlit which produces predicted risk of MDD along with its driving factors. The analysis concluded all four hypotheses were supported.

## Data Preprocessing
The original data contained 4184 rows and 62 columns, representing 4184 students and their 62 attributes ranging from EHR (e.g., blood pressure), demographics (e.g., “Gender”) to lifestyle (e.g., “Eating junk food”).  The preview to the right shows some of the features for 5 students. Among the 62 features, most are categorical (e.g., Eating Junk Food) with only 9 numeric features (eg., “Weight”).. Missing values were observed for some features (e.g., NaN for “Overweight and obesity”). Additional steps were taken to impute missing values and transform the categorical features so that they can be used in the model training.

To help better understand the data, both summary statistics and univariate/bivariate distributions for all features were created. A correlation analysis was also conducted and visualized using Cramer's V heatmap visualizations. 

## Feature Engineering
**Missing Value Imputation:** For numeric features, missing values were imputed with the mean of the feature while median and mode can also be used. An indicator feature was created accordingly with 1 indicating the value was imputed. For categorical features, missing values were replaced by “Unknown”.

**Categorical Encoding:** This includes methods like One-Hot Encoding (OHE) and Ordinal Encoding (OE). It is the technique used to encode categorical features into numerical values so that they can be understood by algorithms. OHE is where each value of a categorical feature is represented by a dummy indicator feature with 1’s and 0’s, while in ordinal encoding, each unique category value is assigned an integer value. The original categorical columns were removed, and the newly encoded columns were appended to the dataset. Whether to choose OHE or OE depends on the data and algorithms. Linear algorithms can only take OHE, while nonlinear algorithms can take both. In this project, a copy of each was created so both linear and nonlinear algorithms can be modeled.

**Feature Creation:** Creation of new features based on existing features often relies on domain knowledge. For instance, height and weight are highly correlated with each other; for this particular project, the interaction of the two might be more important; therefore, a new feature, “BMI_eng,” was created. The distribution of BMI_eng can be found on the left, which seems to be highly correlated with “Overweight and obesity”.

# Feature Selection
## Univariate Feature Selection
This approach selects features based on statistical tests such as Pearson correlation and Chi2 score. In this project, Chi2 score was calculated for each feature; features were then ranked by the score from high to low, and the top features were retained.

One limitation: it only accounts for the relationship between a single feature and the target, ignoring the interactions among features. Therefore, it is often used more conservatively with the judgment based on domain knowledge.

To facilitate univariate feature selection, a noise feature was created and ranked with all other features. The ranking of the noise feature provides guidance as to which features could potentially be dropped.

In this project, only features with a Chi2 score below 0.05 (about 12 features) were removed; all the other features will be further explored using multivariate approach.

## Mulivariate Feature Selection
As compared with the univariate approach, this approach has the advantage of incorporating the interactions among features. Among the many methods, this project chose to leverage both tree-based feature importance and permutation-based feature importance.

Tree-based feature importance measures the importance of a feature based on the mean reduction of impurity with the feature. 

Permutation-based feature importance is defined as the decrease in model accuracy by shuffling (or randomizing) a feature.

Note: the noise feature was ranked as most important by a Random Forest Tree Classifier, which seemed to suggest no features were deemed important in this project. It turned out that this model was poor in performance. Some hyperparameter tuning mitigated the situation. This is a manifestation of the limitation of this approach: it depends on the quality of a model. A feature deemed as of low importance in a bad model could be important in a good model. Therefore, the best practice is to base the decision on several approaches along with domain knowledge.
Combining the results from both approaches, the top 12 features were retained as the final list of features. 

# Model Evaluation
Four algorithms were trained: Logistic Regression, Random Forest, Gradient Boost Machine, and XGBoost. The first three models used sklearn, while XGBoost used the Python library of xgboost. 
Each model was trained on the training data; metrics such as Logloss, AUC, Accuracy, and F1 Score were produced.

**The selected model:** Random Forest was trained with the following hyperparameters: max_depth = 4, max_features = 4,  n_estimators = 1000, criterion = “gini”

### Observations:
Since the data only has ~4000 rows, there is no big difference in model training time. With that said, the Logistic Regression model took the least time to run since it has the fewest parameters; while the other three tree-based models could take much longer with a low learning rate or a high number of estimators.
Both linear and nonlinear models have similar performance as measured by the four metrics. This indicates advanced algorithms don’t necessarily have an advantage here, probably because the features in this data are relatively simple. Therefore, neural networks and deep learning models were not explored in this project.
Random Forest Classifier seems to perform the best with the highest AUC and accuracy, so this model was chosen. Additional evaluations of this model (Lift Chart, ROC) were also performed.

# Conclusions
The analysis confirmed that MDD was correlated with multiple variables from EHR, demographics and personal lifestyles. Not surprisingly, anxiety symptoms turned out to be one of the most important features in all the models explored, showing the potential of using the diagnosis of anxiety to assist the detection of depression. Insights from the machine learning models such as feature importance and prediction explanations facilitated the understanding of drivers for depression in adolescents. 

**Limitations and Future Improvements:** Data is the key for a machine learning model. It works the best when it is used to make predictions on data similar to the training data. The training data used in this project was from college students at Nice University, which makes the model built less applicable for other age groups or other geographical region.
Unstructured text data from social media could help improve the model's performance significantly. It was not considered in this project due to the efforts needed to for data collection. 
With additional data types, more advanced algorithms such as neural networks, deep learning could be explored to further improve the model accuracy.

# Web Application
An interactive Streamlit app `streamlit_app.py` uses the XGBoost model trained on the dataset to predict the likelihood of MDD in adolescents. It uses the data that the user inputs -- demographic and electronic health record (EHR) data and displays a comprehensive summary of factors that are highly correlated and less correlated with MDD. 

The application features:
- **Home page** explaining the context and motivation of the project
- **Exploratory Data Analysis (EDA) page** for exploration of the dataset used to train the model
- **Prediction page** where users input their own data and receive a personalized prediction with SHAP-based model explainability

# Tools Used
- Excel, Python (IDE: Jupyter Notebook)
- Python libraries
  - pandas, numpy for data manipulation
  - sklearn, scipy, xgboost, shap for feature selection and model training, evaluation
  - seaborn, matplotlib for plotting and visualization
  - streamlit for app building

