"""
===============================================================================
Title:          AlzhAI
Description:    This script performs detection of Alzheimer's disease using 
                statistical data.

Author:         Daniele De Carli
Email:          danieledecarli04@gmail.com
Date Created:   2024-08-30
Last Modified:  2024-09-01
Version:        1.0.0
===============================================================================
"""

#Library for data
import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

#Neural Network library
import keras
from keras.models import Sequential
from keras.layers import Dense
import preprocessing_datas


path = 'archive/trained_model.keras'

def diagnose_new_patient(new_patient_data, path):

    # Convert new_patient_data to DataFrame if it's a dictionary
    if isinstance(new_patient_data, dict):
        new_patient_df = pd.DataFrame([new_patient_data])
    else:
        new_patient_df = new_patient_data
    
    new_patient_df_standard_scaled, _ = preprocessing_datas.processing_data(new_patient_df)

    new_patient_df_standard_scaled = new_patient_df_standard_scaled.drop(["DoctorInCharge", "PatientID"], axis=1)
    new_patient_df_standard_scaled = new_patient_df_standard_scaled.astype(np.float32)

    # Model
    model = Sequential()  # Initialize a Sequential model
    model.add(Dense(units=1, activation='relu', input_shape=(new_patient_df_standard_scaled.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])
    
    model.load_weights(path)
    
    # Prediction
    prediction = model.predict(new_patient_df_standard_scaled)
    
    # Convert the prediction to binary outcome
    diagnosis = 1 if prediction >= 0.5 else 0
    
    return diagnosis

# In the following example expected Diagnosis = 1 = Alzheimer's diseas
new_patient_data = {
    "PatientID": "4768",
    "Age": 65,
    "Gender": 1,
    "Ethnicity": 0,
    "EducationLevel": 1,	
    "BMI": 16.33328,
    "Smoking": 1,	
    "AlcoholConsumption": 4.161795,	 
    "PhysicalActivity": 1.30632,	
    "DietQuality": 2.888936,	
    "SleepQuality":	5.436423,
    "FamilyHistoryAlzheimers": 0,	
    "CardiovascularDisease": 0,
    "Diabetes": 0,
    "Depression": 0,	
    "HeadInjury": 0,
    "Hypertension": 1,
    "SystolicBP": 154,	
    "DiastolicBP": 61,	
    "CholesterolTotal": 183.1123,	
    "CholesterolLDL": 101.2582,	
    "CholesterolHDL": 39.22966601,	
    "CholesterolTriglycerides": 374.8551646,	
    "MMSE": 18.04929389,	
    "FunctionalAssessment": 4.019546237,
    "MemoryComplaints": 0,
    "BehavioralProblems": 0,	
    "ADL": 2.892939936,	
    "Confusion": 0,	
    "Disorientation": 0,	
    "PersonalityChanges": 0,	
    "DifficultyCompletingTasks": 0,	
    "Forgetfulness": 0,
    "DoctorInCharge": "XXXConfid",
}

diagnosis = diagnose_new_patient(new_patient_data, path)
print("\n\n\nALZAI VERSION 1 - GitHub: Deca04 - Daniele De Carli")
print("//------------------------//------------------------//")
print(f"Predicted diagnosis for the new patient: {'Alzheimer' if diagnosis == 1 else 'No Alzheimer'}")
print("//------------------------//------------------------//\n\n\n")
