### section 1: let's import some libraries

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app

import shap #for prediction explanation
import streamlit.components.v1 as components

### end of section 1 ###

### Section 2: let's define some functions
@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value
@st.cache(suppress_st_warning=True)
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

#get SHAP plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

### end of section 2 ###


### section 3: read in some of the data that's needed for different pages

# read in the raw data for exploratory data analysis
df = pd.read_excel("database(well-being of students in Nice).xls")
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

#lload the training data
X_nonlinear_train = pd.read_pickle("df_nonlinear_train.pkl")

### end of section 3 ###


### section 4: create the layout and fill in each page with relavant information

## subsection 4.1: create three pages that can be toggled from one to another on the sidebar
app_mode = st.sidebar.selectbox('Select Page',['Home',"Exploratory Data Analysis", 'Prediction']) #three pages

## subsection 4.2: page "Home"
if app_mode=='Home':
    st.sidebar.subheader('Early Detection of Major Depressive Disorder in Adolescents Using Demographics and Electronic Health Records')  
    st.image('depression-hub.jpeg')
    st.markdown("Mental health is an integral component of the overall health of humans, yet many are not being treated for mental health disordes. Major depressive disorder (MDD) is a chronic, common disorder among adolescents, with lifetime recurrence rates of ~70%. MDD has been on the rise in recent years, impacting adolescents particularly with a 63% increase from 2013 to 2016. This situation was only worsened with COVID, with a 25% increase in depression worldwide. However, early signs of depression often go unnoticed among adolescents; moreover fears of stigmatization and social rejection and concerns of lack of confidentiality impose barriers for them to seek help from adults and primary care providers. For many individuals, especially adolescents, major depression can result in severe impairments that interfere with or limit their ability to carry out major life activities, such as learning at school. High school students around the world are particularly vulnerable to mental health issues while facing unprecedented challenges imposed by fast changing technology and social and political environment.")
    st.markdown("Therefore early detection of major depression among high school students is of paramount importance. .... This study aims to apply various machine learning algorithms on demographics and electronic health records with the goal of detecting major depression at early stage. The results of the analysis can potentially provide insights into how schools can proactively involve with students with potential depressive disorders.")



## subsection 4.3: page "Exploratory Data Analysis
elif app_mode == "Exploratory Data Analysis":
    st.subheader('Exploratory Data Analysis')  
    st.caption('Dataset Preview')
   
#     df = pd.read_excel("database(well-being of students in Nice).xls")
    st.write(df.head())
   
    st.caption('Distribution of each feature in the data')
   
    fig=plt.figure(figsize = (3, 2))
    var_selected = st.selectbox("Please select a feature:", cols)
    if var_selected in num_cols:
        sns.histplot(data = df, x = var_selected, bins = 15)
        st.pyplot(fig)
    elif var_selected in cat_cols:
        sns.countplot(data = df, x = var_selected)
        st.pyplot(fig)
       
## subsection 4.4: page "prediction"    
elif app_mode == 'Prediction':
#     st.image('prediction.jpeg')

    st.subheader('Please answer the questions on the left and then click the Predict button below')
    st.sidebar.header("Informations about the student:")
   
    # define the dictionaries for each categorical varibles - to map actual categorical values to encoded numerical values
    dict_Difficulty_memorizing_lessons = {'no':0, 'yes':1}
    dict_Anxiety_symptoms = {'no':0, 'yes':1}
    dict_Physical_activity = {'no':0, 'occasionally':1, 'regularly':2}
    dict_Satisfied_with_living_conditions = {'no':0, 'unknown':1, 'yes':2}
    dict_Financial_difficulties = {'no':0, 'yes':1}
    dict_Learning_disabilities = {'no':0, 'yes':1}
    dict_Having_only_one_parent = {'no':0, 'unknown':1, 'yes':2}
    dict_Unbalanced_meals = {'no':0, 'yes':1}
    dict_Eating_junk_food = {'no':0, 'yes':1}
    dict_Cigarette_smoker = {'frequently':0, 'heavily':1, 'no':2, 'occasionally':3, 'regularly':4, 'unknown':5}
   
    # create the inputs for each variable in the model on the sidebar so users can either selection or type in their answers
    Difficulty_memorizing_lessons = st.sidebar.radio("Difficulty memorizing lessons?", tuple(dict_Difficulty_memorizing_lessons.keys()))
    Anxiety_symptoms = st.sidebar.radio("Anxiety symptoms?", tuple(dict_Anxiety_symptoms.keys()))
    Height_Imputed = st.sidebar.number_input("Height (cm)")
    Physical_activity = st.sidebar.radio("Physical activity", tuple(dict_Physical_activity.keys()))
    Satisfied_with_living_conditions = st.sidebar.radio("Satisfied with living_conditions?", tuple(dict_Satisfied_with_living_conditions.keys()))
    BMI_eng_imputed = st.sidebar.number_input("Body Mass Index (BMI)")
    HeartRate_imputed = st.sidebar.number_input("Heart Rate")
    Financial_difficulties = st.sidebar.radio("Financial difficulties?", tuple(dict_Financial_difficulties.keys()))
    Learning_disabilities = st.sidebar.radio("Learning_disabilities?", tuple(dict_Learning_disabilities.keys()))
    Having_only_one_parent = st.sidebar.radio("Having only one parent?", tuple(dict_Having_only_one_parent.keys()))
    Unbalanced_meals = st.sidebar.radio("Unbalanced meals?", tuple(dict_Unbalanced_meals.keys()))
    Eating_junk_food = st.sidebar.radio("Eating junk food?", tuple(dict_Eating_junk_food.keys()))
    Cigarette_smoker = st.sidebar.radio("Cigarette smoker?", tuple(dict_Cigarette_smoker.keys()))
   

    # save the inputs collected in data1
    data1={
       'Difficulty memorizing lessons' : Difficulty_memorizing_lessons,
       'Anxiety symptoms' : Anxiety_symptoms,
       'Height Imputed' : Height_Imputed,
       'Physical activity(3 levels)' : Physical_activity,
       'Satisfied with living conditions' : Satisfied_with_living_conditions,
       'BMI_eng imputed' : BMI_eng_imputed,
       'HeartRate_imputed' : HeartRate_imputed,
       'Financial difficulties' : Financial_difficulties,
       'Learning disabilities':Learning_disabilities,
       'Having only one parent':Having_only_one_parent,
       'Unbalanced meals':Unbalanced_meals,
       'Eating junk food':Eating_junk_food,
       'Cigarette smoker (5 levels)' : Cigarette_smoker
       }

    feature_list=[Height_Imputed,
             HeartRate_imputed,
             BMI_eng_imputed,
             get_value(Financial_difficulties, dict_Financial_difficulties),
             get_value(Eating_junk_food, dict_Eating_junk_food),
             get_value(Physical_activity, dict_Physical_activity),
             get_value(Cigarette_smoker, dict_Cigarette_smoker),
             get_value(Having_only_one_parent, dict_Having_only_one_parent),
             get_value(Anxiety_symptoms, dict_Anxiety_symptoms),
             get_value(Learning_disabilities, dict_Learning_disabilities),
             get_value(Satisfied_with_living_conditions, dict_Satisfied_with_living_conditions),
             get_value(Unbalanced_meals, dict_Unbalanced_meals),
             get_value(Difficulty_memorizing_lessons, dict_Difficulty_memorizing_lessons)
            ]        

    single_sample = np.array(feature_list).reshape(1,-1)



    if st.button("Predict"):
#     file_ = open("images_pass.jpeg", "rb")
#     contents = file_.read()
#     data_url = base64.b64encode(contents).decode("utf-8")
#     file_.close()

#     file = open("images_warning.jpeg", "rb")
#     contents = file.read()
#     data_url_no = base64.b64encode(contents).decode("utf-8")
#     file.close()

        pickled_model = pickle.load(open('model_xgb.pkl', 'rb'))
#         st.write(single_sample)
        prediction = pickled_model.predict_proba(single_sample)
       
        col1, col2 = st.columns(2)
        if prediction[:,1] < 0.5:
            col1.metric(label = "Model predicted probability of depression", value = np.round(prediction[:,1], 2))
            col2.metric(label = 'Depression Risk', value = "Low")
    #         st.image('images_pass.jpeg')
        else:
            col1.metric(label = "Model predicted probability of depression", value = np.round(prediction[:,1], 2))
            col2.metric(label = 'Prediction', value = "High")
    #         st.image('images_warning.jpeg')


        explainer = shap.TreeExplainer(pickled_model)
        shap.initjs()
        single_sample = pd.DataFrame(single_sample, columns = X_nonlinear_train.columns)
        shap_values = explainer.shap_values(single_sample)
#         st.write(single_sample)
       
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#         st_shap(shap.force_plot(explainer.expected_value, shap_values, X_nonlinear_train.columns, link="logit"))
       
        st.set_option('deprecation.showPyplotGlobalUse', False)
       
        shap_value_local = explainer(single_sample)
        shap.waterfall_plot(shap_value_local[0])
        st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
        plt.clf()
       
### end of section 4 ###
