import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def import_datas():
    df = pd.read_csv("archive/alzheimers_disease_data.csv")
    print(df.head())
    df.shape
    df.info()
    df.dtypes
    return df

sns.set_theme()
plt.figure(figsize=(14, 6))  #Figure dimensions

def age_historgams(df):
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='Age', kde=True)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age distribution')

    mean_age = df['Age'].mean()
    plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean Age: {mean_age:.1f}')

    
    df_positive = df[df['Diagnosis'] == 1] #filter of df
    age_counts = df_positive['Age'].value_counts()

    #histogram for positive diagnosis
    plt.subplot(1, 2, 2)
    sns.barplot(x=age_counts.index, y=age_counts.values)

    plt.xlabel('Age')
    plt.ylabel('Number of Positive Diagnoses')
    plt.title('Positive Diagnoses by Age')

    plt.show()


def correletation_matrix(): 
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

    #Heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    plt.title('Correlation Matrix of Health and Diagnostic Variables', fontsize=20, fontweight='bold')

    plt.grid(False)
    plt.show()

#Called functions
df = import_datas()
age_historgams(df)
correletation_matrix()
