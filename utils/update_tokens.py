########################################################################################

# cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################

import pandas as pd
import re
import json
import argparse

########################################################################################

def smiles_segmenter(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smi = smi.replace(" ", "")
    tokens = [token for token in regex.findall(smi)]
    smi_new = "".join(tokens)
    # if smi != smi_new:
        # print(f"\n{smi}\n{smi_new}")
    # assert smi == smi_new
    return tokens

########################################################################################

def update_token_dict(dataset_loc, main_token_dict_path, args):
    df = pd.read_csv(dataset_loc)
    with open(main_token_dict_path, "r") as f:    
        main_token_dict = json.load(f)
    ####################################################
    new_token_dict = {}
    for i in range(len(df)):
        smi = df[args.x_label][i]
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

def main(args):
    ####################################################
    data_names = args.data_names.replace(" ", "")
    data_names = data_names.split(",")
    ####################################################
    for data_name in data_names:
        args.data_name = data_name
        new_dataset_all_loc = f"{args.data_folder}/{args.data_name}/{args.data_name}_all.csv"
        update_token_dict(new_dataset_all_loc, args.tokens_path, args)
        print(f"\n{args.data_name}")
    ####################################################

########################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--x_label", default="smiles", type=str)
    parser.add_argument("--data_folder", default="data", type=str)
    parser.add_argument("--data_names", default="", type=str)
    parser.add_argument("--data_name", default="", type=str)   # sonra değer atanıyor
    parser.add_argument("--tokens_path", default="CHARSET.json", type=str)
    ##########################
    args = parser.parse_args()
    ##########################    
    return args

########################################################################################

if __name__ == "__main__":
    ##########################
    args = load_args()
    args.tokens_path = args.data_folder + "/" + args.tokens_path
    main(args)
    ##########################

########################################################################################




        

































