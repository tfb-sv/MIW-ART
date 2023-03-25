cd C:\Users\nural\OneDrive\Masaüstü\MARTree_fixCHA

#########################################################################################################

from pubchempy import *
import pandas as pd
import time
import os
import json

#########################################################################################################
# CREATE DICT FOR THE FIRST TIME

td1 = {}
loc0 = "z.ex/n2_datasets/csvs/"
cnt = 0
for file_name in os.listdir(loc0):
    file_loc = loc0 + file_name
    print(file_name)
    df = pd.read_csv(file_loc)
    for i in range(len(df)):
        smi = df["smiles"][i]
        td1[smi] = cnt
        cnt += 1
        
# print(len(td1))

with open("smi2cid.json", "w") as f:
    json.dump(td1, f)

#########################################################################################################
# UPDATE THE DICT 

with open("data/smi2cid.json", "r") as f:
    td1 = json.load(f)

loc0 = "new_bbbp/"
cnt = 97819
for file_name in os.listdir(loc0):
    file_loc = loc0 + file_name
    print(file_name)
    df = pd.read_csv(file_loc)
    for i in range(len(df)):
        smi = df["smiles"][i]
        if smi not in td1:
            td1[smi] = cnt
            cnt += 1
            
with open("data/smi2cid2.json", "w") as f:
    json.dump(td1, f)

#########################################################################################################

# td2 = {}
# cnt = 0
# start = time.time()
# for smi in td1:
#     try:
#         c = get_compounds(smi, 'smiles')
#         if len(c) > 1:
#             print("what?", c)
#             break
#         c = c[0]
#         new_smi = c.canonical_smiles
#         cid = c.cid
#     except:
#         print("\n", smi, "bulunamadı..")
#         new_smi = smi
#         c = 0
#         cid = cnt
#     td2[new_smi] = [cid, c, cnt]
#     print(cnt)
#     cnt += 1
# end = time.time()
# print(((end - start) / 60), "mins")
        
#########################################################################################################
#########################################################################################################




















