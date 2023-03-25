##########################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

##########################################################################

import pandas as pd
import numpy as np

##########################################################################

def check_data_is_same(task_lst, split_lst):
    ##################################
    print("\n##################################")
    ##################################
    for task in task_lst:
        for split in split_lst:
            ##################################
            file_loc = f"data/{task}/{task}_{split}.csv"
            file_loc_new = f"data_new/{task}/{task}_{split}.csv"
            ##################################
            df = pd.read_csv(file_loc)
            df_new = pd.read_csv(file_loc_new)
            ##################################
            tmp_lst = list(df["smiles"])
            cnt = 0
            for i in range(len(df_new)):
                smi_new = df_new["smiles"][i]
                if smi_new in tmp_lst:
                    cnt += 1
            ##################################
            ratio1 = np.round(((cnt / len(df_new)) * 100), 2)
            ratio2 = np.round(((cnt / len(df)) * 100), 2)
            ##################################
            print(f"\n>> {task.upper():7s} {split.capitalize():5s} : {ratio1} % NEW : {ratio2} % OLD")
            ##################################
        print("\n##################################")
    print("\n")
    ##################################
    return

##########################################################################
    
task_lst = ["bbbp2k", "bbbp8k", "bace", "clintox", "tox21", "lipo"]
split_lst = ["train", "val", "test", "all"]

##################################

check_data_is_same(task_lst, split_lst)

##########################################################################



































