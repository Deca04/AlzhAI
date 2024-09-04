"""
===============================================================================
Title:          AlzhAI
Description:    This script performs detection of Alzheimer's disease using 
                statistical data.

Author:         Daniele De Carli & Alberto Sudati
Email:          danieledecarli04@gmail.com    ; alberto.sudati04@gmail.com
Date Created:   2024-09-04
Last Modified:  2024-09-04
Version:        1.0.2
===============================================================================
"""

#________________________Library__________________________________________#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, "C:\\Users\\YUOR_NAME\\Desktop\\work in progress\\Progetti\\ALZAI\\Code\\ALZAI statistical approach\\custom_libraries")
import preprocessing_datas

#_________________________________________________________________________#

file_csv = 'archive/alzheimers_disease_data.csv'

df = preprocessing_datas.read_csv(file_csv)
df_standard_scaled, df_minmax_scaled = preprocessing_datas.processing_data(df)    

print(df_standard_scaled.head())
print(df_minmax_scaled.head())

df = df.drop("DoctorInCharge", axis=1) 
df = df.drop("PatientID", axis=1) 

X = df.drop("Diagnosis", axis=1) 
y = df["Diagnosis"] 

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)  

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

cicli1 = 40  # Number of iterations for n_estimators
cicli2 = 10  # Number of iterations for max_depth
sfasamento1 = 105
sfasamento2 = 12

accuracy_array = np.zeros((cicli1, cicli2))

for i in range(cicli1):
    for j in range(cicli2):
        clf = RandomForestClassifier(
            n_estimators=i+sfasamento1,          
            max_depth=j+sfasamento2,                
            max_features='sqrt',          
            criterion='gini',             
            min_samples_split=2,          
            min_samples_leaf=1,           
            bootstrap=True,               
            random_state=42               
        )
            
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        accuracy_array[i, j] = acc

        print(f"n_estimators: {i+sfasamento1}, max_depth: {j+sfasamento2} => Accuracy: {acc:.4f}")


# Plot 3D graph
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(sfasamento1, sfasamento1+cicli1), np.arange(sfasamento2, sfasamento2+cicli2))
ax.plot_surface(X, Y, accuracy_array.T, cmap='viridis')

ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('Accuracy')

plt.title("Accuracy in function of n_estimators and max_depth")
plt.show()
