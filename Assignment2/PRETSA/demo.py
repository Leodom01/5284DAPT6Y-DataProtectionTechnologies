# In order to get the cleaned sepsis csv file (sepsis_shrinked_duration.csv)
# it's required to run xasToCsv.py, dataset_shrinker.py and add_annotation_duration.py
import sys
sys.path.append("../PRETSA")
import utils as utils
from pretsa import Pretsa
import pandas as pd
from anytree.exporter import DotExporter

filePath="/Users/leodom01/Repos/DataProtectionTechnologies/Assignment2/PRETSA/datasets/GoodEnvironmental/env_duration.csv"
min_k = int(sys.argv[1])
max_t = float(sys.argv[2])


pd_dataset = pd.read_csv(filePath, delimiter=";")
pretsaNoSaviour = Pretsa(pd_dataset)
pretsaWithSaviour = Pretsa(pd_dataset)

print("K values BEFORE: ", pretsaWithSaviour.getKvalues())
cutOutCases = pretsaWithSaviour.runPretsa(min_k,max_t)
privateEventLog = pretsaWithSaviour.getPrivatisedEventLog()
privateEventLog.to_csv("/Users/leodom01/Repos/DataProtectionTechnologies/Assignment2/PRETSA/datasets/GoodEnvironmental/env_duration_sanitized.csv", sep=";",index=False)

print("K values AFTER quick way: ", pretsaWithSaviour.getKvalues())

