# Reduce the number of columns to a minimum
import pandas as pd


def shrinkSepsis(columns_to_keep=['org:group', 'Hypotensie', 'Age', 'Diagnose', 'Activity', 'time:timestamp', 'Case ID']):
    pd_df = pd.read_csv('datasets/Sepsis cases/sepsis_enriched.csv', delimiter=";")
    pd_df = pd_df[columns_to_keep]
    pd_df.to_csv('datasets/Sepsis cases/sepsis_shrinked.csv', sep=";", index=False)

# TODO: Implement this also for the 2 other datasets

if __name__ == "__main__":
    shrinkSepsis()