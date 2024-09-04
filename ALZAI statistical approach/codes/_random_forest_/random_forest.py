"""
===============================================================================
Title:          AlzhAI
Description:    This script performs detection of Alzheimer's disease using 
                statistical data.

Author:         Daniele De Carli & Alberto Sudati
Email:          danieledecarli04@gmail.com    ; alberto.sudati04@gmail.com
Date Created:   2024-09-03
Last Modified:  2024-09-04
Version:        1.0.0
===============================================================================
"""

#________________________Library__________________________________________#

import numpy as np

import sys
sys.path.insert(0, "C:\\Users\\YUOR_NAME\\Desktop\\work in progress\\Progetti\\ALZAI\\Code\\ALZAI statistical approach\\custom_libraries")
import preprocessing_datas

import matplotlib.pyplot as plt

# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#_________________________________________________________________________#

file_csv = 'archive/alzheimers_disease_data.csv'

df = preprocessing_datas.read_csv(file_csv)
df_standard_scaled, df_minmax_scaled = preprocessing_datas.processing_data(df)    

print(df_standard_scaled.head())
print(df_minmax_scaled.head())

df = df.drop("DoctorInCharge", axis=1)    # Drop the target column to get the features
df = df.drop("PatientID", axis=1)         # Drop the target column to get the features

# Split the data into features (X) and target (y)
X = df.drop("Diagnosis", axis=1) 
y = df["Diagnosis"] 

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)  
#random state controls how the data is shuffled before the split is implemented

print('X_train:',np.shape(X_train))
print('y_train:',np.shape(y_train))
print('X_test:',np.shape(X_test))
print('y_test:',np.shape(y_test))

# Convert features to float
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Convert target to int (if not already)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

cicli1 = 20
cicli2 = 10

accuracy_array = np.zeros((cicli1*cicli2,2))

for i in range(cicli1):
    for j in range(cicli2):
        clf = RandomForestClassifier(
            n_estimators=i+100,           
            max_depth=j+5,               
            max_features='sqrt',          
            criterion='gini',            
            min_samples_split=2,          
            min_samples_leaf=1,           
            bootstrap=True,               
            random_state=42               
        )
            
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        mcc = matthews_corrcoef(y_test, y_pred)  
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        print("ACCURACY OF THE MODEL:", acc)


        conf_mat = confusion_matrix(y_test, y_pred)
        displ = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        print(f"Classifier - Confusion Matrix:\n{conf_mat}")

        accuracy_array[j+cicli2*i,0] = j+cicli2*i
        accuracy_array[j+cicli2*i,1] = acc

print(accuracy_array)

plt.figure(figsize=(8, 6))
plt.plot(accuracy_array[:,1], label='Accuracy vs n_estimators', marker='x')

plt.title("Accuracy over n-trees")
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.grid(True)

plt.show()
