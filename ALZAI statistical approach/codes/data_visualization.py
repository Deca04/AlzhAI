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

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme()

plt.figure(figsize=(14, 6))  # Dimensione della figura

#import datas
df = pd.read_csv("archive/alzheimers_disease_data.csv")
df.shape
df.info()
df.dtypes

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Age', kde=True)
plt.title('Age distribution')

df_positive = df[df['Diagnosis'] == 1] #filter of df
age_counts = df_positive['Age'].value_counts().sort_index()

#histogram for positive diagnosis
plt.subplot(1, 2, 2)
sns.barplot(x=age_counts.index, y=age_counts.values)

plt.xlabel('Age')
plt.ylabel('Number of Positive Diagnoses')
plt.title('Positive Diagnoses by Age')

plt.show()

plt.figure(figsize=(50, 10))

# Calculate the correlation matrix
corr_matrix = df[["Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
                  "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
                  "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
                  "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
                  "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
                  "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
                  "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
                  "Forgetfulness", "Diagnosis"]].corr()

# Create the heatmap with improved formatting
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Add the title with improved formatting
plt.title('Correlation Matrix of Health and Diagnostic Variables', fontsize=20, fontweight='bold')

# Display the plot
plt.grid(False)
plt.show()
