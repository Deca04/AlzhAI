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

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: impossible to find the file '{file_path}'.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file'{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Parsing error in the file'{file_path}'.")
        return None
    except Exception as e:
        print(f"Error during file reading: {e}")
        return None

def missing_data(df):
    # Checking for blank data
    blank_data = df.isnull().sum()
    print("Blank data by column:")
    print(blank_data)

    # Checking for NaN data
    nan_data_by_column = df.isna().sum()
    nan_total_data = df.isna().sum().sum()

    print("NaN data by column:")
    print(nan_data_by_column)
    print("\nTotal NaN data in DataFrame:", nan_total_data)

    # Checking for missing data
    missing_data = df.isnull()

    print("Missing data by column:")

    # Deleting columns
    df.drop(['PatientID'],axis=1,inplace=True)
    df.drop(['DoctorInCharge'],axis=1,inplace=True)

    # Print the total number of missing values in the entire DataFrame.
    total_missing = df.isnull().sum().sum() 
    print("Total data missing in DataFrame:", total_missing)


'''
    Normalization transforms the data such that the minimum value of each feature becomes 0 and the maximum value becomes 1. 
    This process is particularly useful when:

    *** Consistent Scale Across Features: Normalization ensures that all features have the same scale, which can be important for algorithms that compute 
        distances between data points, such as K-Means clustering or K-Nearest Neighbors (KNN).
    *** Maintaining the Relationship: Unlike standardization, normalization maintains the relative relationships between the values. 
        This is useful when the distribution of data needs to be preserved.
    *** Improving Convergence: For optimization algorithms that rely on gradient descent (like neural networks), 
        normalization can improve the convergence speed and overall performance.
'''


def processing_data(df):
    if df is None:
        print("No DataFrame provided for data processing.")
        return None, None
    
    # Print DataFrame info to debug the structure
    #print("DataFrame structure before processing:")
    #print(df.info())
    #print("\nFirst few rows of the DataFrame:")
    #print(df.head())

    # Initializing StandardScaler
    scaler_standard = StandardScaler()

    # Applying StandardScaler to variables, correcting the error with reshape
    df['Age'] = scaler_standard.fit_transform(df['Age'].values.reshape(-1, 1))
    df['BMI'] = scaler_standard.fit_transform(df['BMI'].values.reshape(-1, 1))
    df['AlcoholConsumption'] = scaler_standard.fit_transform(df['AlcoholConsumption'].values.reshape(-1, 1))
    df['PhysicalActivity'] = scaler_standard.fit_transform(df['PhysicalActivity'].values.reshape(-1, 1))
    df['DietQuality'] = scaler_standard.fit_transform(df['DietQuality'].values.reshape(-1, 1))
    df['SleepQuality'] = scaler_standard.fit_transform(df['SleepQuality'].values.reshape(-1, 1))
    df['SystolicBP'] = scaler_standard.fit_transform(df['SystolicBP'].values.reshape(-1, 1))
    df['DiastolicBP'] = scaler_standard.fit_transform(df['DiastolicBP'].values.reshape(-1, 1))
    df['CholesterolTotal'] = scaler_standard.fit_transform(df['CholesterolTotal'].values.reshape(-1, 1))
    df['CholesterolLDL'] = scaler_standard.fit_transform(df['CholesterolLDL'].values.reshape(-1, 1))
    df['CholesterolHDL'] = scaler_standard.fit_transform(df['CholesterolHDL'].values.reshape(-1, 1))
    df['CholesterolTriglycerides'] = scaler_standard.fit_transform(df['CholesterolTriglycerides'].values.reshape(-1, 1))
    df['MMSE'] = scaler_standard.fit_transform(df['MMSE'].values.reshape(-1, 1))
    df['FunctionalAssessment'] = scaler_standard.fit_transform(df['FunctionalAssessment'].values.reshape(-1, 1))
    df['ADL'] = scaler_standard.fit_transform(df['ADL'].values.reshape(-1, 1))

    # Initializing MinMaxScaler
    scaler_minmax = MinMaxScaler()

    # Applying MinMaxScaler to variables, correcting the error with reshape
    df['Age'] = scaler_minmax.fit_transform(df['Age'].values.reshape(-1, 1))
    df['BMI'] = scaler_minmax.fit_transform(df['BMI'].values.reshape(-1, 1))
    df['AlcoholConsumption'] = scaler_minmax.fit_transform(df['AlcoholConsumption'].values.reshape(-1, 1))
    df['PhysicalActivity'] = scaler_minmax.fit_transform(df['PhysicalActivity'].values.reshape(-1, 1))
    df['DietQuality'] = scaler_minmax.fit_transform(df['DietQuality'].values.reshape(-1, 1))
    df['SleepQuality'] = scaler_minmax.fit_transform(df['SleepQuality'].values.reshape(-1, 1))
    df['SystolicBP'] = scaler_minmax.fit_transform(df['SystolicBP'].values.reshape(-1, 1))
    df['DiastolicBP'] = scaler_minmax.fit_transform(df['DiastolicBP'].values.reshape(-1, 1))
    df['CholesterolTotal'] = scaler_minmax.fit_transform(df['CholesterolTotal'].values.reshape(-1, 1))
    df['CholesterolLDL'] = scaler_minmax.fit_transform(df['CholesterolLDL'].values.reshape(-1, 1))
    df['CholesterolHDL'] = scaler_minmax.fit_transform(df['CholesterolHDL'].values.reshape(-1, 1))
    df['CholesterolTriglycerides'] = scaler_minmax.fit_transform(df['CholesterolTriglycerides'].values.reshape(-1, 1))
    df['MMSE'] = scaler_minmax.fit_transform(df['MMSE'].values.reshape(-1, 1))
    df['FunctionalAssessment'] = scaler_minmax.fit_transform(df['FunctionalAssessment'].values.reshape(-1, 1))
    df['ADL'] = scaler_minmax.fit_transform(df['ADL'].values.reshape(-1, 1))

    # Applying StandardScaler and MinMaxScaler to the 'Age' column
    df_standard_scaled = df.copy()
    df_minmax_scaled = df.copy()

    # Example of comparing the original, standardized and rescaled versions of a variable
    #print("Original:", df['Age'].head())
    #print("Standard Scaled:", df_standard_scaled['Age'].head())
    #print("MinMax Scaled:", df_minmax_scaled['Age'].head())

    return df_standard_scaled, df_minmax_scaled

if __name__ == "__main__":
    file_csv = 'archive/alzheimers_disease_data.csv'

     # Check if the file exists before reading
    if not os.path.isfile(file_csv):
        print(f"Error: File '{file_csv}' not found.")
    else:
        df = read_csv(file_csv)
        if df is not None:
            missing_data(df)
            df_standard_scaled, df_minmax_scaled = processing_data(df)
    
        

       
