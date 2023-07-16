import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        gender = st.sidebar.radio('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 0, 20, 100)
        hypertension = st.sidebar.radio('Has Hypertension', (1, 0))
        heart_disease = st.sidebar.radio('Has Heart Disease', (1, 0))
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

df = df.iloc[:1]

#transform data
with open('scalerNorm.pkl', 'rb') as f:
    scaler = pickle.load(f)
scaled_df = scaler.transform(df)



# Reads in saved classification model

load_clf = pickle.load(open('hgbc_model_lda_pickle.pkl', 'rb'))
# Apply model to make predictions
prediction = load_clf.predict(scaled_df)
prediction_proba = load_clf.predict_proba(scaled_df)

with st.container():
    st.write('-----')
    left_column, right_column =st.columns(2)
    with left_column:
        st.header('Prediction')
        stroke_type = np.array([0, 1])
        if stroke_type[prediction] == 1:
            prediction_stroke = "You have a high risk of getting a future stroke"
        else:
            prediction_stroke = "You have a low risk of getting a future stroke"
        st.write(prediction_stroke)
        
        st.subheader('Prediction Probability')
        #creates a piechart showing probability
        labels = ['Low risk of stroke', 'High risk of stroke']
        sizes = [prediction_proba[0,0],prediction_proba[0,1]]
        colors = ['red', 'yellow']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        fig.set_facecolor('gray')
        st.pyplot(fig)


