# -*- coding: utf-8 -*-
"""
Created on Thursday March-5 2026
@author: Ashish Acharya
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu


# ================= LOAD MODELS =================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# ================= SIDEBAR =================
with st.sidebar:

    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )


# ================= DIABETES PREDICTION =================
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', 0, 20)

    with col2:
        Glucose = st.number_input('Glucose Level', 0, 300)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value', 0, 200)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value', 0, 100)

    with col2:
        Insulin = st.number_input('Insulin Level', 0, 900)

    with col3:
        BMI = st.number_input('BMI value', 0.0, 70.0)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', 0.0, 2.5)

    with col2:
        Age = st.number_input('Age of the Person', 1, 120)

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):

        input_data = np.array([[Pregnancies, Glucose, BloodPressure,
                                SkinThickness, Insulin, BMI,
                                DiabetesPedigreeFunction, Age]])

        diab_prediction = diabetes_model.predict(input_data)

        if diab_prediction[0] == 1:
            diab_diagnosis = '⚠ The person is diabetic'
        else:
            diab_diagnosis = '✅ The person is not diabetic'

    st.success(diab_diagnosis)


# ================= HEART DISEASE PREDICTION =================
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using Machine Learning')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', 1, 120)

    with col2:
        sex = st.selectbox('Sex', [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    with col3:
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200)

    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', 100, 600)

    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")

    with col1:
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2])

    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', 60, 220)

    with col3:
        exang = st.selectbox('Exercise Induced Angina', [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes")

    with col1:
        oldpeak = st.number_input('ST Depression (oldpeak)', 0.0, 10.0)

    with col2:
        slope = st.selectbox('Slope of ST Segment', [0, 1, 2])

    with col3:
        ca = st.selectbox('Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])

    with col1:
        thal = st.selectbox('Thal', [0, 1, 2, 3])

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):

        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach, exang,
                                oldpeak, slope, ca, thal]])

        heart_prediction = heart_disease_model.predict(input_data)

        if heart_prediction[0] == 1:
            heart_diagnosis = '⚠ The person has Heart Disease'
        else:
            heart_diagnosis = '✅ The person does not have Heart Disease'

    st.success(heart_diagnosis)


# ================= PARKINSONS PREDICTION =================
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using Machine Learning")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', 0.0)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', 0.0)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', 0.0)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', 0.0)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', 0.0)

    with col1:
        RAP = st.number_input('MDVP:RAP', 0.0)

    with col2:
        PPQ = st.number_input('MDVP:PPQ', 0.0)

    with col3:
        DDP = st.number_input('Jitter:DDP', 0.0)

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', 0.0)

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', 0.0)

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', 0.0)

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', 0.0)

    with col3:
        APQ = st.number_input('MDVP:APQ', 0.0)

    with col4:
        DDA = st.number_input('Shimmer:DDA', 0.0)

    with col5:
        NHR = st.number_input('NHR', 0.0)

    with col1:
        HNR = st.number_input('HNR', 0.0)

    with col2:
        RPDE = st.number_input('RPDE', 0.0)

    with col3:
        DFA = st.number_input('DFA', 0.0)

    with col4:
        spread1 = st.number_input('spread1', 0.0)

    with col5:
        spread2 = st.number_input('spread2', 0.0)

    with col1:
        D2 = st.number_input('D2', 0.0)

    with col2:
        PPE = st.number_input('PPE', 0.0)

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):

        input_data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs,
                                RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                                APQ3, APQ5, APQ, DDA, NHR, HNR,
                                RPDE, DFA, spread1, spread2, D2, PPE]])

        parkinsons_prediction = parkinsons_model.predict(input_data)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "⚠ The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "✅ The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
