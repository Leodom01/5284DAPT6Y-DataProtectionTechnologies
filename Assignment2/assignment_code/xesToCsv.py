import pandas as pd
import pm4py
import pandas as pd

# I am aware it's not the best parsing, but it's quick and clear

# Load the XES file
sepsis_log = pm4py.read_xes(
    "../PRETSA/datasets/Sepsis cases/Sepsis Cases - Event Log.xes")
environmental_log = pm4py.read_xes(
    "../PRETSA/datasets/Environmental permit application/CoSeLoG WABO 1.xes")
traffic_log = pm4py.read_xes(
    "../PRETSA/datasets/Road traffic management dataset/Road_Traffic_Fine_Management_Process.xes")


# Convert the XES sepsis_log to a DataFrame
dataframe = pm4py.convert_to_dataframe(sepsis_log)
# I manually decided which attribute uniquely represent the cases
dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
# Save the DataFrame to a CSV file
dataframe.to_csv('../PRETSA/datasets/Sepsis cases/sepsis.csv', index=False)

# Convert the XES environmental_log to a DataFrame
dataframe = pm4py.convert_to_dataframe(environmental_log)
# I manually decided which attribute uniquely represent the cases
dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
# Save the DataFrame to a CSV file
dataframe.to_csv('../PRETSA/datasets/Environmental permit application/environmental.csv', index=False)

# Convert the XES traffic_log to a DataFrame
dataframe = pm4py.convert_to_dataframe(traffic_log)
# I manually decided which attribute uniquely represent the cases
dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
# Save the DataFrame to a CSV file
dataframe.to_csv('../PRETSA/datasets/Road traffic management dataset/traffic.csv', index=False)

