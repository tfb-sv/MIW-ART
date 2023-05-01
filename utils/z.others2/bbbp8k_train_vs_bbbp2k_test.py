##############################################################################
##############################################################################
##############################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

##############################################################################

import pandas as pd
import copy

##############################################################################
##############################################################################
##############################################################################
# Use the test and valid set of BBBP2K exactly, but for the train set use for the new created unified set which is alls_c.

def load_df(data_name, set_name):
    f_lock = f"data/bbbp{data_name}/bbbp{data_name}_{set_name}.csv"
    df = pd.read_csv(f_lock)
    return df

#####################################################

df2_train = load_df("2k", "train")
df2_valid = load_df("2k", "val")
df2_test = load_df("2k", "test")
df8_all = load_df("8k", "all")

##############################################################################

alls = pd.concat([df2_train, df8_all], axis=0, ignore_index=True)
alls = alls.drop('Unnamed: 0', axis=1)
alls_c = copy.deepcopy(alls)
        
##############################################################################

duplicates_valid = df2_valid[df2_valid["smiles"].isin(alls_c["smiles"])]
duplicates_test = df2_test[df2_test["smiles"].isin(alls_c["smiles"])]

#####################################################

alls_c = alls_c[~alls_c["smiles"].isin(duplicates_valid["smiles"])]
alls_c = alls_c[~alls_c["smiles"].isin(duplicates_test["smiles"])]

#####################################################

alls_c = alls_c.drop_duplicates(subset="smiles", keep="first")

alls_c.to_csv("alls_c.csv")

##############################################################################
##############################################################################
##############################################################################












































