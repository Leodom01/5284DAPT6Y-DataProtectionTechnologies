import pandas as pd
import pm4py
import pandas as pd

# I am aware it's not the best parsing, but it's quick and clear

def xesToCsvDataset(datasetName="sepsis"):

    if datasetName == "sepsis":
        # Load the XES file
        sepsis_log = pm4py.read_xes(
            "datasets/Sepsis cases/Sepsis Cases - Event Log.xes")
        # Convert the XES sepsis_log to a DataFrame
        dataframe = pm4py.convert_to_dataframe(sepsis_log)
        # I manually decided which attribute uniquely represent the cases
        dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
        dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
        # Save the DataFrame to a CSV file
        dataframe.to_csv('datasets/Sepsis cases/sepsis.csv', sep=";", index=False)
    elif datasetName == "environment":
        environmental_log = pm4py.read_xes(
            "datasets/Environmental permit application/CoSeLoG WABO 1.xes")
        # Convert the XES environmental_log to a DataFrame
        dataframe = pm4py.convert_to_dataframe(environmental_log)
        # I manually decided which attribute uniquely represent the cases
        dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
        dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
        # Save the DataFrame to a CSV file
        dataframe.to_csv('datasets/Environmental permit application/environmental.csv', sep=";", index=False)
    elif datasetName == "traffic":
        traffic_log = pm4py.read_xes(
            "datasets/Road traffic management dataset/Road_Traffic_Fine_Management_Process.xes")
        # Convert the XES traffic_log to a DataFrame
        dataframe = pm4py.convert_to_dataframe(traffic_log)
        # I manually decided which attribute uniquely represent the cases
        dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
        dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
        # Save the DataFrame to a CSV file
        dataframe.to_csv('datasets/Road traffic management dataset/traffic.csv', sep=";", index=False)
    elif datasetName == "goodEnv":
        env_log = pm4py.read_xes(
            "datasets/GoodEnvironmental/CoSeLoG_duration.xes")
        # Convert the XES traffic_log to a DataFrame
        dataframe = pm4py.convert_to_dataframe(env_log)
        # I manually decided which attribute uniquely represent the cases
        dataframe.rename(columns={'case:concept:name': 'Case ID'}, inplace=True)
        dataframe.rename(columns={'concept:name': 'Activity'}, inplace=True)
        # Save the DataFrame to a CSV file
        dataframe.to_csv('datasets/GoodEnvironmental/env.csv', sep=";", index=False)


if __name__ == "__main__":
    # Running this script standalone will convert all the datasets available
    # xesToCsvDataset("sepsis")
    # xesToCsvDataset("environment")
    # xesToCsvDataset("traffic")
    xesToCsvDataset("goodEnv")