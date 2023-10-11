# Reduce the number of columns to a minimum
import pandas as pd

columns_to_keep = ['org:group', 'Activity', 'Age', 'time:timestamp', 'Case ID',
                   'InfectionSuspected', 'DiagnosticBlood']
pd_df = pd.read_csv('../PRETSA/datasets/Sepsis cases/sepsis.csv', delimiter=";")
pd_df = pd_df[columns_to_keep]
pd_df.to_csv('../PRETSA/datasets/Sepsis cases/sepsis_shrinked.csv', sep=";", index=False)

# TODO: Implement this also for the 2 other datasets