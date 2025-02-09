import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('src/model/model.pkl')

# Title of the web app
st.title('Heart Disease Detection')

# Create a form for user input
def user_input_features():
    Age = st.slider('Age', min_value=18, max_value=80, value=30)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Blood_Pressure = st.slider('Blood Pressure',  min_value=50, max_value=250, value=150)
    Cholesterol = st.slider('Cholesterol Level', min_value=100, max_value=350, value=225)
    Exercise_Habits = st.selectbox('Exercise Habits', ['High', 'Medium', 'Low'])
    Smoking = st.selectbox('Smoking', ['Yes', 'No'])
    Family_Heart_Disease = st.selectbox('Family Heart Disease', ['Yes', 'No'])
    Diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
    BMI = st.slider('BMI', min_value=10, max_value=50, value=30)
    High_Blood_Pressure = st.selectbox('High Blood Pressure', ['Yes', 'No'])
    Low_HDL_Cholesterol = st.selectbox('Low HDL Cholesterol', ['Yes', 'No'])
    High_LDL_Cholesterol = st.selectbox('High LDL Cholesterol', ['Yes', 'No'])
    Alcohol_Consumption = st.selectbox('Alcohol Consumption', ['Medium', 'Low', 'High'])
    Stress_Level = st.selectbox('Stress Level', ['Low', 'Medium', 'High'])
    Sleep_Hours = st.slider('Sleep Hours', min_value=0, max_value=24, value=7)
    Sugar_Consumption = st.selectbox('Sugar Consumption', ['Low', 'Medium', 'High'])
    Triglyceride_Level = st.slider('Triglyceride Level', min_value=50, max_value=500, value=250)
    Fasting_Blood_Sugar = st.slider('Fasting Blood Sugar', min_value=50, max_value=200, value=120)
    CRP_Level = st.slider('CRP Level', min_value=0.0001, max_value=16.0, value=7.5)
    Homocysteine_Level = st.slider('Homocysteine Level', min_value=0.1, max_value=30.0, value=15.0)

    # Create a DataFrame with standardized feature names
    features = pd.DataFrame([{
        'Age': Age,
        'Gender': 1 if Gender == 'Male' else 0,
        'Blood_Pressure': Blood_Pressure,
        'Cholesterol': Cholesterol,
        'Exercise_Habits': 2 if Exercise_Habits == 'High' else (1 if Exercise_Habits == 'Medium' else 0),
        'Smoking': 1 if Smoking == 'Yes' else 0,
        'Family_Heart_Disease': 1 if Family_Heart_Disease == 'Yes' else 0,
        'Diabetes': 1 if Diabetes == 'Yes' else 0,
        'BMI': BMI,
        'High_Blood_Pressure': 1 if High_Blood_Pressure == 'Yes' else 0,
        'Low_HDL_Cholesterol': 1 if Low_HDL_Cholesterol == 'Yes' else 0,
        'High_LDL_Cholesterol': 1 if High_LDL_Cholesterol == 'Yes' else 0,
        'Alcohol_Consumption': 2 if Alcohol_Consumption == 'High' else (1 if Alcohol_Consumption == 'Medium' else 0),
        'Stress_Level': 2 if Stress_Level == 'High' else (1 if Stress_Level == 'Medium' else 0),
        'Sleep_Hours': Sleep_Hours,
        'Sugar_Consumption': 2 if Sugar_Consumption == 'High' else (1 if Sugar_Consumption == 'Medium' else 0),
        'Triglyceride_Level': Triglyceride_Level,
        'Fasting_Blood_Sugar': Fasting_Blood_Sugar,
        'CRP_Level': CRP_Level,
        'Homocysteine_Level': Homocysteine_Level
    }])

    return features

# Get user input
input_data = user_input_features()

# Ensure the column names match the model's expected input features
expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_data.columns
input_data = input_data.reindex(columns=expected_columns, fill_value=0)  # Ensure all expected columns are present

# Make predictions
prediction = model.predict(input_data)



# Display results
st.subheader('Prediction:')
st.write("Heart Disease Detected" if prediction == 'Yes' else "No Heart Disease Detected")
