import pandas as pd
from collections import Counter
import sys

fileToCheck = sys.argv[1]

# Load the dataset
df = pd.read_csv(fileToCheck, delimiter=";")
counter = Counter()
# Iterate over unique Case IDs

for case_id in df['Case ID'].unique():

    current_case = df[df['Case ID'] == case_id]

    currentAnnotation = ""

    for index, value in current_case['Activity'].items():
        currentAnnotation = currentAnnotation+"@"+value[:5]

    if currentAnnotation in counter.keys():
        counter[currentAnnotation] += 1
    else:
        counter[currentAnnotation] = 1

print("Sanitized counter:")
for value, count in counter.most_common():
    print(value+ " : "+str(count))