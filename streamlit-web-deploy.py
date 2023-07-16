import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

import requests
from streamlit_lottie import st_lottie

st.set_theme('Green')
# checks if url is accessible
def load_lottieurl():
    r = requests.get(url)
    if r.status_code != 200:
        return None 
    return r.json()
# load animations assets
animation_assets = "https://lottie.host/bba4dee9-1aca-4f0f-877c-bd6dabe45d9a/2eECIRnNXa.json"
animation_assets2 = "https://lottie.host/c6d26031-6c6e-4fcd-af9c-9ca061b99002/TM8rXPAdPB.json"
animation_assets3 = "https://lottie.host/9918e8b0-ede8-4566-b114-d2d2c0210649/OVYpu4whI1.json"

st.title('Welcome To My Stroke Prediction Website  :wave:')
st.write('###')
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/jceu/stroke/main/example_data.csv)
""")

#collect user input features into a dataframe

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.radio('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 0, 20, 100)
        hypertension = st.sidebar.radio('Has Hypertension', ("Yes", "No"))
        heart_disease = st.sidebar.radio('Has Heart Disease', ("Yes", "No"))
        ever_married = st.sidebar.radio('Ever Married', ('Yes', 'No'))
        work_type = st.sidebar.selectbox('Type of Work', ('Private', 'Self-employed','Govt_job','children','Never_worked'))
        residence_type = st.sidebar.radio('Type of Residence', ('Urban','Rural'))
        avg_glucose_level = st.sidebar.slider('Average Glucose Level', 50, 20, 100)
        bmi = st.sidebar.slider('BMI Level', 10, 20, 55)
        smoking_status = st.sidebar.selectbox('Smoking Status', ('formerly smoked', 'never smoked','smokes','Unknown'))
        data = {'gender': gender,
               'age': age,
               'hypertension': hypertension,
               'heart_disease': heart_disease,
               'ever_married': ever_married,
               'work_type': work_type,
               'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status,
               }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# shows user inputs
st.subheader('User Input Features')
st.write('#####')
if uploaded_file is not None:
    st.table(input_df)
else:
    st.table(input_df)
    

#combine user input features with entire dataset

stroke_df = pd.read_csv('adjusted_dataset_stroke.csv')

stroke_df_dropped = stroke_df.drop(columns=['stroke'])

df = pd.concat([input_df, stroke_df_dropped], axis=0)

# perform one-hot-encoding on 'gender' and 'home' columns

encode = ['gender','work_type','smoking_status','Residence_type']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Replace the categories with the boolean values    
label_mapping_married = {'No': False, 'Yes': True}
df['ever_married'] = df['ever_married'].map(label_mapping_married)

label_mapping_hypertension = {'No': 0, 'Yes': 1}
df['hypertension'] = df['hypertension'].map(label_mapping_hypertension)

label_mapping_heart_disease = {'No': 0, 'Yes': 1}
df['heart_disease'] = df['heart_disease'].map(label_mapping_heart_disease)

df = df.iloc[:1]

#transform data
with open('scalerNorm.pkl', 'rb') as f:
    scaler = pickle.load(f)
scaled_df = scaler.transform(df)

with st.container():
    st.write('-----') 
    left_column, right_column =st.columns(2,gap="medium")
    with right_column:
        st_lottie(animation_assets, height = 600)
    with left_column:
        st.write("##")
        st.subheader("Data Dictionary")
        data_dictionary = {
        "gender": "Gender of user Male or Female",
        "age": "Age of user",
        "hypertension": "Whether user has hypertension",
        "heart_disease": "Whether user has heart disease",
        "ever_married": "Whether user has been married before",
        "work_type": "The type of work",
        "Residence_type": "The type of residence",
        "avg_glucose_level": "average glucose level of user",
        "bmi": "Body mass index (bmi) of user",
        "smoking_status": "The smoking status of user"
        }
        st.table(data_dictionary)


# Reads in saved classification model
load_clf = pickle.load(open('hgbc_model_lda_pickle.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(scaled_df)
prediction_proba = load_clf.predict_proba(scaled_df)

# relavent information of prediction displayed
with st.container():
    st.write('---')
    left_column, right_column = st.columns(2,gap="large")
    
    with left_column:
        st.write("#")
        st.write("#")
        st.subheader('Prediction Probability')
        #creates a piechart showing probability
        labels = ['Low risk of stroke', 'High risk of stroke']
        sizes = [prediction_proba[0,0],prediction_proba[0,1]]
        colors = ["#8B0000", "#008000"]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={"color": "white"})
        ax.axis('equal')
        fig.set_facecolor('#0B0C10')
        fig.set_size_inches(3, 3)
        st.pyplot(fig)
        st.write("(Open web page in wide mode to view pie chart better and have a better overall experience)")
        
    with right_column:
        st.header('Prediction: ')
        stroke_type = np.array([0, 1])
        if stroke_type[prediction] == 1:
            prediction_stroke = "You have a high risk of getting a future stroke, try to eat healthy and exercise more!"
            st_lottie(animation_assets2, height = 600)
        else:
            prediction_stroke = "You have a low risk of getting a future stroke, continue to exercise and eat healthy!"
            st_lottie(animation_assets3, height = 600)
        st.subheader(prediction_stroke)
        
        

