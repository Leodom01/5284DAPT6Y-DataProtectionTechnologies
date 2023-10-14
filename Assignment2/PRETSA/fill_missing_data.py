import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/leodom01/Repos/DataProtectionTechnologies/Assignment2/PRETSA/datasets/Sepsis cases/sepsis.csv')
headers = df.iloc[0]
# Iterate over unique Case IDs

total_cases = df['Case ID'].nunique()
checked_cases = 0

for case_id in df['Case ID'].unique():
    checked_cases += 1
    print("Checked ", 100*checked_cases/total_cases, "%")
    case_df = df[df['Case ID'] == case_id]  # Select entries for a specific Case ID

    for column in df.columns:
        truth_val = None
        for index, value in case_df[column].items():
            if not pd.isna(value):
                truth_val = value  # Set truth_val to the non-NaN value
            elif truth_val is not None:
                df.at[index, column] = truth_val  # Fill NaN with truth_val

    # Update the original DataFrame with the filled values
    df.update(case_df)

# Save the modified DataFrame to a new CSV file
df.to_csv('/Users/leodom01/Repos/DataProtectionTechnologies/Assignment2/PRETSA/datasets/Sepsis cases/sepsis_enriched.csv', index=False)

print("Completed data enhanching")