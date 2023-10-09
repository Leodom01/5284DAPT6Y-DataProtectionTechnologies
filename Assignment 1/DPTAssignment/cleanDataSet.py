import pandas as pd

#The dataset has been collected from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

# Load dataset from CSV
df = pd.read_csv('sleep_health_and_lifestyle_dataset.csv')

# Remove the 'Person ID' field
df = df.drop(columns=['Person ID'])

# Transform the 'Gender' field into a numeric one (1 = female, 0 = male)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Remove the 'Occupation' field
df = df.drop(columns=['Occupation'])

# Create a mapping for the 'BMI Category' field
bmi_mapping = {
    'Underweight': 0,
    'Normal': 1,
    'Overweight': 2,
    'Obese': 3
}
df['BMI Category'] = df['BMI Category'].map(bmi_mapping)

# Split the 'Blood Pressure' field into 'Systolic' and 'Diastolic'
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df = df.drop(columns=['Blood Pressure'])

# Transform the 'Sleep Disorder' field into a boolean field (1 if there's a disorder, 0 if None)
df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: 1 if x != 'None' else 0)

df['Quality of Sleep'] = df['Quality of Sleep'].apply(lambda x: 1 if x > 7.5 else 0)

# Convert the processed DataFrame to a CSV file
df.to_csv('processed_dataset.csv', index=False)

