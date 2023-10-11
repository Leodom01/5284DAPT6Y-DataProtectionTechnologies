import pandas as pd

# Define the columns to keep
columns_to_keep = ['org:group', 'Activity', 'Age', 'time:timestamp', 'Case ID',
                   'InfectionSuspected', 'DiagnosticBlood', 'Duration']

# Read the CSV data into a DataFrame
df = pd.read_csv('sepsis_duration.csv', delimiter=";")

# Select the desired columns
df = df[columns_to_keep]

# Save the filtered DataFrame to a new CSV file
df.to_csv('sepsis_shrinked_duration.csv', index=False)
