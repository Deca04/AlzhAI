#Library for data
import numpy as np
import preprocessing_datas 

#Library fro trainign
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

#Neural Network library
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from keras.layers import Dense


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

file_csv = 'archive/alzheimers_disease_data.csv'

df = preprocessing_datas.read_csv(file_csv)
df_standard_scaled, df_minmax_scaled = preprocessing_datas.processing_data(df)    

print(df_standard_scaled.head())
print(df_minmax_scaled.head())

df = df.drop("DoctorInCharge", axis=1) # Drop the target column to get the features
df = df.drop("PatientID", axis=1) # Drop the target column to get the features

# Split the data into features (X) and target (y)
X = df.drop("Diagnosis", axis=1) # Drop the target column to get the features
y = df["Diagnosis"] # Set the target column

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

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

complete_matrix = np.zeros((10,2))



for i in range (0,10):
    #INITIALIZE THE SEQUENTIAL MODEL
    basic_model = Sequential()                                                           
    basic_model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))  
    basic_model.add(Dense(32, activation='relu'))
    basic_model.add(Dense(64, activation='relu'))
    basic_model.add(Dense(16, activation='relu'))
    basic_model.add(Dense(1, activation='sigmoid'))                                      

    adam = keras.optimizers.Adam(learning_rate=0.001)                                    

    basic_model.compile(loss='binary_focal_crossentropy', optimizer=adam, metrics=["accuracy"])


    history = basic_model.fit(X_train, y_train, epochs=20, batch_size=32)

    loss_and_metrics = basic_model.evaluate(X_test, y_test)
    print(loss_and_metrics)
    print('Loss = ',loss_and_metrics[0])
    print('Accuracy = ',loss_and_metrics[1])

    predicted = basic_model.predict(X_test)

    predicted = tf.squeeze(predicted)
    predicted = np.array([1 if x >= 0.5 else 0 for x in predicted])
    actual = np.array(y_test)
    conf_mat = confusion_matrix(actual, predicted)
    
    complete_matrix[i,0] = conf_mat[1,0]
    complete_matrix[i,1] = conf_mat[0,1]
    
print(complete_matrix)

path = 'archive/trained_model_v3.keras'

basic_model.summary()
#basic_model.save(path)   #save the model in a .keras file
basic_model.save_weights(path)