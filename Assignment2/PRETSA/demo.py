"""
    This demo will prove the advantages in using the K saviour during the pretsa sanitization.
    Here's a short overview of the main features implemented, more details can be found in the docstrings in pretsa.py

    Pretsa would delete (then modify and merge in a different branch) all the prefix tree nodes that do not follow the
    t-closeness and k-anonymity constraint.
    K saviour aims to avoid deleting and losing important data by generating synthetic data to add when the anonymity
    set size of a set is slightly below the k-anonymity threshold (the self.__synthEnrichmentThreshold field in the
    pretsa object defines what's the minimum set size for a node not to be deleted (and then merged) but to be
    enriched with synthetic data.

    All the annotation field data is generated through the enhanced __generateNewAnnotation that implements a more
    accurate way of generating annotations, inspired by Taylor Series.
    The original way of generating annotation data was based on sampling data from a natural distribution whose
    specs originated by looking at all the annotations (durations) of the activities with the same name.
    This method is not totally correct since the duration of a log is the time that passed between the previous and the
    current action, therefore not taking into consideration the previous action could generate a highly improbable
    result.
    The enhance annotation generation algorithm takes samples from a bimodal distribution where one distribution is
    generated as in the classic algorithm, the second distribution is generated from logs that have the same activity
    name AND have the same activity name in the previous log. The second distribution has a double weight compared to
    the first one (the classical one). In this way we generate a data starting from a dataset that is much closely
    related to our target (the log we have to generate the annotation for).
"""

import sys
sys.path.append("../PRETSA")
from pretsa import Pretsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filePath="/Users/leodom01/Repos/DataProtectionTechnologies/Assignment2/PRETSA/datasets/GoodEnvironmental/env_duration.csv"
max_t = 0.8

k_values = np.arange(2, 21, 2)

# Difference between added and removed/modified (lost) logs
log_delta_saviour = []
log_delta_std = []
# Difference between added and removed/modified (lost) cases
cases_delta_saviour = []
cases_delta_std = []
# Number of distinct sequences available in the dataset (to show how various the data is)
distinct_sequence_saviour = []
distinct_sequence_std = []

# Run one sanitization for multiple Ks to show the behaviour
# Both of the demo use the enhanced annotationData generator, it's quite hard to quickly show the benefit of a better
# annotation generation, this aspect is explained in the README
print("Welcome to the demo, due to the grat number of sanitization made (about 20), this run will take a couple of minutes.")
for k in k_values:
    print(f"K={k} ( {100*k/20}% )")

    pd_dataset = pd.read_csv(filePath, delimiter=";")
    pretsaStd = Pretsa(pd_dataset, k_saviour=False)
    pretsaWithSaviour = Pretsa(pd_dataset, k_saviour=True)

    # Runnin sanitizatiion with the K saviour
    print("Pretsa with K Saviour feature activated")
    cutOutCases = pretsaWithSaviour.runPretsa(k,max_t)
    privateEventLog = pretsaWithSaviour.getPrivatisedEventLog()
    # Get number of added and lost logs
    addedLogs, removedLogs = pretsaWithSaviour.getAlteredLogs()
    log_delta_saviour.append(addedLogs-removedLogs)
    print(f"Number of logs added and removed/modified (during tree pruning): {addedLogs} {removedLogs}")
    # Get number of added and lost cases
    addedCases, removedCases = pretsaWithSaviour.getAlteredCases()
    cases_delta_saviour.append(addedCases-removedCases)
    print(f"Number of cases added and removed/modified (during tree pruning): {addedCases} {removedCases}")
    # Get number of distinct sequences
    distinctSequences = pretsaWithSaviour.getNumberOfDifferentSequences()
    distinct_sequence_saviour.append(distinctSequences)
    print(f"Number of distinct sequences among all nodes: {distinctSequences}")

    # Now run the standard pretsa to see the difference
    print("Pretsa without K Saviour feature")
    cutOutCases = pretsaStd.runPretsa(k, max_t)
    privateEventLog = pretsaStd.getPrivatisedEventLog()
    # Get number of added and lost logs
    addedLogs, removedLogs = pretsaStd.getAlteredLogs()
    log_delta_std.append(addedLogs - removedLogs)
    print(f"Number of logs added and removed/modified (during tree pruning): {addedLogs} {removedLogs}")
    # Get number of added and lost cases
    addedCases, removedCases = pretsaStd.getAlteredCases()
    cases_delta_std.append(addedCases - removedCases)
    print(f"Number of cases added and removed/modified (during tree pruning): {addedCases} {removedCases}")
    # Get number of distinct sequences
    distinctSequences = pretsaStd.getNumberOfDifferentSequences()
    distinct_sequence_std.append(distinctSequences)
    print(f"Number of distinct sequences among all nodes: {distinctSequences}")

plt.plot(k_values, log_delta_saviour, marker='o', linestyle='-', label='With K saviour')
plt.plot(k_values, log_delta_std, marker='o', linestyle='-', label='Without K saviour')
plt.title('Log variation')
plt.xlabel('Minimum K')
plt.ylabel('Number of logs')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(k_values, cases_delta_saviour, marker='o', linestyle='-', label='With K saviour')
plt.plot(k_values, cases_delta_std, marker='o', linestyle='-', label='Without K saviour')
plt.title('Cases variation')
plt.xlabel('Minimum K')
plt.ylabel('Number of cases')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(k_values, distinct_sequence_saviour, marker='x', linestyle='-', label='With K saviour')
plt.plot(k_values, distinct_sequence_std, marker='x', linestyle='-', label='Without K saviour')
plt.title('Distinct sequences (series of event with different activities)')
plt.xlabel('Minimum K')
plt.ylabel('Number of distinct activity sequences')
plt.grid(True)
plt.legend()
plt.show()