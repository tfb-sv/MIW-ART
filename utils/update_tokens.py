cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################

import pandas as pd
import re
import json

########################################################################################

def smiles_segmenter(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smi = smi.replace(" ", "")
    tokens = [token for token in regex.findall(smi)]
    smi_new = "".join(tokens)
    if smi != smi_new:
        print(f"\n{smi}\n{smi_new}")
    assert smi == smi_new
    return tokens

########################################################################################

def update_token_dict(dataset_loc, main_token_dict_path):
    # dataset_loc = "data/hiv/hiv_all.csv"
    # main_token_dict_path = "data/CHARSET.json"
    df = pd.read_csv(dataset_loc)
    with open(main_token_dict_path, "r") as f:    
        main_token_dict = json.load(f)
    ####################################################
    new_token_dict = {}
    for i in range(len(df)):
        smi = df["smiles"][i]
        tokens = smiles_segmenter(smi)
        for token in tokens:
            if token not in new_token_dict:
                new_token_dict[token] = 0
    ####################################################
    cnt = 0
    print("\n")
    for token in new_token_dict:
        if token not in main_token_dict:
            cnt += 1
            main_token_dict[token] = len(main_token_dict)
            print(f'>> Token "{token}" is added to main dict!  <<')
    print(f"\n>> {cnt} tokens are added to main dict!  <<\n")
    ####################################################
    with open(main_token_dict_path, "w") as f:
        json.dump(main_token_dict, f)
    ####################################################
    return

########################################################################################

datasets = ["bace", "bbbp2k", "clintox", "tox21", "lipo", "esol"]
main_token_dict_path = "data_new/CHARSET.json"

for task_name in datasets:
    new_dataset_all_loc = f"data_new/{task_name}/{task_name}_all.csv"
    update_token_dict(new_dataset_all_loc, main_token_dict_path)
    print(f"\n{task_name}")

########################################################################################








        















