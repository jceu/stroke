import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

st.title('Stroke Prediction Website')

st.sidebar.header('User Input Features')


st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

#collect user input features into a dataframe

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 0, 20, 100)
        hypertension = st.sidebar.selectbox('Has Hypertension', (1, 0))
        heart_disease = st.sidebar.selectbox('Has Heart Disease', (1, 0))
        ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
        work_type = st.sidebar.selectbox('Type of Work', ('Private', 'Self-employed','Govt_job','children','Never_worked'))
        residence_type = st.sidebar.selectbox('Type of Residence', ('Urban','Rural'))
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
    df = user_input_features()

# perform one-hot-encoding on 'gender' and 'home' columns

encode = ['gender','work_type','smoking_status','Residence_type']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Replace the categories with the boolean values    
label_mapping_married = {'No': False, 'Yes': True}
df['ever_married'] = df['ever_married'].map(label_mapping_married)

# transform data
norm = MinMaxScaler().fit(df)
df = norm.transform(df)
 
    
    
    
# select only the first row (the user input data)

# Displays the user input features
st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model

load_clf = pickle.load(open('hgbc_model_lda_pickle.pkl', 'rb'))

# Apply model to make predictions

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
stroke_type = np.array([1, 0])
st.write(stroke_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
