cd C:\Users\nural\OneDrive\Masaüstü\MARTree_fixCHA

#########################################################################################################

from pubchempy import *
import pandas as pd
import time
import os

#########################################################################################################

td1 = {}
loc0 = "z.ex/n2_datasets/csvs/"
cnt = 0
for file_name in os.listdir(loc0):
    file_loc = loc0 + file_name
    print(file_name)
    df = pd.read_csv(file_loc)
    for i in range(len(df)):
        smi = df["smiles"][i]
        td1[smi] = [cnt]
        cnt += 1
        
# print(len(td1))

#########################################################################################################
















