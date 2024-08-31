# AlzhAI

## Project Overview
This project aims to develop an inclusive algorithm for the detection, recognition, and prediction of Alzheimer's disease through the integration of statistical-historical data and MRI images. The goal is to create a reliable model that can assist in the early diagnosis and prediction of dementia-related diseases such as Alzheimer's.

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

   Note: For a comprehensive explanation of each variable, please refer to the previous link to the dataset, where all variables are detailed thoroughly.
   
   ### 2. Development of Predictive Model
   After identifying the key correlations from the historical data (through a correlation matrix), the next step involves:

   * Model Design: Developing a predictive model using Tensorflow.
     A basic model turns out to be efficient enough for the moment, an architecture built with one input and one output layer appears to minimize the loss function with an average accuracy of 82 percent.
   * Model Training: once the model is trained on 2150 patients' data, the weight coefficients are save and used in a main example program where an example patient is tested.
