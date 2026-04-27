import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler

#load model
model = pickle.load(open('rf_model.pkl','rb'))

#title for app
st.title('Heart Attack Risk Classification App')

#create input features

age = st.number_input('Age',min_value = 20, max_value = 100, value=25)
restingbp = st.number_input('Resting BP',min_value = 0, max_value = 300, value=100)
cholesterol = st.number_input('Cholesterol',min_value = 0, max_value = 700, value=140)
fastingbs = st.selectbox('Fasting BS',['0','1'])
maxhr = st.number_input('Max HR',min_value = 60, max_value = 250, value=140)
oldpeak = st.number_input('Oldpeak',min_value = -3.0, max_value = 6.6, value=1.0)
gender = st.selectbox('Gender',['Male','Female'])
chestpaintype = st.selectbox('Chest Pain Type',['ATA','NAP','ASY','TA'])
restingecg = st.selectbox('Resting ECG',['Normal','ST','LVH'])
exerciseangina = st.selectbox('Exercise Angina',['No','Yes'])
st_slope = st.selectbox('St_Slope',['Up','Flat','Down'])

#encoding logic
Exercise_Angina = 1 if exerciseangina=='Y' else 0

Sex_F = 1 if gender=='F' else 0
Sex_M = 1 if gender=='M' else 0

Chest_PainType_dict = {'ASY':3,'NAP':2,'ATA':1,'TA':0}
Chest_PainType = Chest_PainType_dict[chestpaintype]

Resting_ECG_dict = {'ST':2,'LVH':1,'Normal':0}
Resting_ECG = Resting_ECG_dict[restingecg]

ST_Slope_dict = {'Flat':2,'Up':1,'Down':0}
ST_Slope = ST_Slope_dict[st_slope]

#create dataframe
input_features = pd.DataFrame({'Age':[age],
                               'RestingBP':[restingbp],
                               'Cholesterol':[cholesterol],
                               'FastingBS':[fastingbs],
                               'MaxHR':[maxhr],
                               'Oldpeak':[oldpeak],
                               'Exercise_Angina':[Exercise_Angina],
                               'Sex_F':[Sex_F],
                               'Sex_M':[Sex_M],
                               'Chest_PainType':[Chest_PainType],
                               'Resting_ECG':[Resting_ECG],
                               'st_Slope':[ST_Slope]})

#scaling
scaler = StandardScaler()
input_features[['Age','RestingBP','Cholesterol','MaxHR']] = scaler.fit_transform(input_features[['Age','RestingBP','Cholesterol','MaxHR']])

#prediction
if st.button('Predict'):
  predictions = model.predict(input_features)[0]
  if predictions ==1:
    st.error('High risk of Heart Attack!')
  else:
    st.success('Low Heart Attack Risk')