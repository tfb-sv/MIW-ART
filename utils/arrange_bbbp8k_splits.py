############################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol\data

############################################################

import pandas as pd
from sklearn.model_selection import train_test_split

############################################################

# file1 = "bbbp2k/bbbp2k_test.csv"
file2 = "bbbp8k/bbbp8k.tsv"

############################################################

# df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2, sep="\t")

############################################################

# tmp = {}
# tst = {}
# for i in range(len(df2)):
#     smi = df2["SMILES"][i]
#     clf = df2["BBB+/BBB-"][i]
#     if clf[-1] == "+":
#         clf = 1
#     elif clf[-1] == "-":
#         clf = 0
#     else:
#         sdlfkjsdf
#     if smi in df1["smiles"]:
#         tst[smi] = clf    
#     else:
#         tmp[smi] = clf
        
############################################################

tmp = {}
for i in range(len(df2)):
    smi = df2["SMILES"][i]
    clf = df2["BBB+/BBB-"][i]
    if clf[-1] == "+":
        clf = 1
    else:
        clf = 0
    tmp[smi] = clf
        
############################################################

x = list(tmp.keys())
y = list(tmp.values())

x_t, x_test, y_t, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
x_t, x_v, y_t, y_v = train_test_split(x_t, y_t, test_size=0.11, random_state=0)

############################################################

clms = ["smiles", "y_true"]

train = pd.concat([pd.DataFrame(x_t), pd.DataFrame(y_t)], axis=1)
train.columns = clms
valid = pd.concat([pd.DataFrame(x_v), pd.DataFrame(y_v)], axis=1)
valid.columns = clms
test = pd.concat([pd.DataFrame(x_test), pd.DataFrame(y_test)], axis=1)
test.columns = clms
allx = pd.concat([pd.DataFrame(train), pd.DataFrame(valid), pd.DataFrame(test)], axis=0)

############################################################

train.to_csv("bbbp8k_train.csv")
valid.to_csv("bbbp8k_val.csv")
test.to_csv("bbbp8k_test.csv")
allx.to_csv("bbbp8k_all.csv")

############################################################





















