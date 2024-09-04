![logo_mid](https://github.com/user-attachments/assets/6a0469e0-80c8-44fc-863f-7939b9eab653)


## Project Overview
This project aims to develop an inclusive algorithm for the detection, recognition, and prediction of Alzheimer's disease through the integration of statistical-historical data and MRI images. The goal is to create a reliable model that can assist in the early diagnosis and prediction of dementia-related diseases such as Alzheimer's.

The project is developed by @Deca04 and @AlbeSud

## Project Structure
   ### 1. Statistical Analysis of Historical Data
   The first phase of the project focuses on the statistical analysis of historical data from patient records. This includes:

   * Data Collection: Gathering patient data, including demographics, medical history, and other relevant features.
   * Preprocessing: Cleaning and organizing the data to ensure quality and consistency.
   * Correlation Analysis: Identifying key features that have the highest correlation with the onset of Alzheimer's disease.
   * Feature Selection: Selecting the most relevant features to be used in the predictive model.

   The dataset used (updated at 2024) is the following: https://www.kaggle.com/dsv/8668279
   The **dataset** contain: 
   * Patient Information
   * Patient ID
   * Demographic Details
   * Lifestyle Factors
   * Medical History
   * Clinical Measurements
   * **Cognitive and Functional Assessments**
   * Symptoms
   * Diagnosis Information

   **Note**: For a comprehensive explanation of each variable, please refer to the previous link to the dataset, where all variables are detailed thoroughly.
   
   ### 2. Development of Predictive Model
   After identifying the key correlations from the historical data (through a correlation matrix), the next step involves:

   * Model Design: Developing a predictive model using Tensorflow.
     A basic model turns out to be efficient enough for the moment, an architecture built with one input and one output layer appears to minimize the loss function with an average accuracy of 82 percent.
   * Model Training: once the model is trained on 2150 patients' data, the weight coefficients are save and used in a main example program where an example patient is tested.

   ### 3. Libraries and Dependencies
   The following Python libraries are used in this project:

   * *Numpy*: For numerical operations and handling arrays.
   * *Matplotlib*: For visualizing data and plotting graphs.
   * *Scikit-learn*: Used for machine learning tasks, including data preprocessing, model training, and evaluation.
   * *TensorFlow and Keras*: Deep learning libraries for building and training neural networks.
   * Custom Library (*preprocessing_datas*): A custom-built module for reading and preprocessing the dataset.

   ### 4. Data Preprocessing
   **Loading and Preparing the Dataset**
   The dataset is loaded from a CSV file (`alzheimers_disease_data.csv`) using the `preprocessing_datas.read_csv` function. Two types of data scaling are applied:

   1. Standard Scaling: Ensures that the features have a mean of 0 and a standard deviation of 1.
   2. Min-Max Scaling: Scales features to a range between 0 and 1.
   Columns DoctorInCharge and PatientID are dropped as they are not relevant for the prediction task. The dataset is then split into features (X) and the target variable (y, which represents the diagnosis).

   **Train-Test Split**
   The dataset is divided into training and testing sets using an 80-20 split:

   ```ruby
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)
   ```
   The split is performed with shuffling to ensure a random distribution of data points.

   ### 5. Neural Network Model
   **Model Architecture**
   The neural network model is built using Keras’ Sequential API. The architecture consists of:

   * Input Layer: Accepts input features.
   * Hidden Layers: Four hidden layers with ReLU activation functions (16, 32, 64, and 16 units, respectively).
   * Output Layer: A single neuron with a sigmoid activation function for binary classification (diagnosis of Alzheimer’s disease).
   * Model Compilation and Training
     
   The model is compiled with:
   - Loss Function: binary_focal_crossentropy to address class imbalance in the dataset.
   - Optimizer: Adam optimizer with a learning rate of 0.001.
   - Metrics: Accuracy is used to evaluate the model.
   The model is trained for 20 epochs with a batch size of 32. The training is repeated 10 times, with the confusion matrix being computed for each iteration to assess the model's performance in terms of false 
   positives and false negatives.

   ```ruby
    for i in range(0,10):
      history = basic_model.fit(X_train, y_train, epochs=20, batch_size=32)
      ...
      complete_matrix[i,0] = conf_mat[1,0]
      complete_matrix[i,1] = conf_mat[0,1]
   ```

   **Model Evaluation and Saving**
   After training, the model's performance is evaluated on the test set, and metrics like loss and accuracy are printed. The model weights are then saved for the `main.py` program.

   ### 6. Random Forest Model
   **Model Configuration**
   A Random Forest classifier is used as an alternative to the neural network model. The model is trained with varying parameters (n_estimators and max_depth) to observe how these hyperparameters affect accuracy.

   **Hyperparameter Tuning**
   The code iterates over a range of values for n_estimators and max_depth to find the optimal configuration. The results are stored in a 2D array (accuracy_array) and then visualized using a 3D surface plot.

```ruby
for i in range(cicli1):
    for j in range(cicli2):
        clf = RandomForestClassifier(n_estimators=i+sfasamento1, max_depth=j+sfasamento2, ...)
        ...
```

**Visualization**
A 3D plot is generated to visualize the relationship between n_estimators, max_depth, and model accuracy.

```ruby
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, accuracy_array.T, cmap='viridis')
```

### 7. Results and Conclusion
The project compares the performance of a neural network and a Random Forest classifier for predicting Alzheimer's disease using patient data. Both models provide insights into the effectiveness of different machine learning approaches in medical diagnostics.

This expanded section in the README file provides users with a thorough understanding of the code structure, the libraries used, and the detailed steps taken for data preprocessing, model building, training, and evaluation.
