############################################################

# cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

############################################################

import deepchem as dc
import pandas as pd
import argparse
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

############################################################

def get_molnet_from_deepchem(args):
    ############################################################ SET THE FOLDERS 
    if not os.path.exists(args.data_folder):   # klasör yoksa, "data" folder
        os.mkdir(args.data_folder, exist_ok=True)   # , exist_ok=True   # klasör oluşturuyor, "training_results" folder
    ##########################
    task_path = (f"{args.data_folder}/{args.ptn})
    ##########################
    if os.path.exists(temp_path):   # klasör varsa, training_results/{task} folder
        shutil.rmtree(temp_path)   # klasör siliyor, training_results/{task} folder
    os.mkdir(task_path)   # klasör oluşturuyor, training_results/{task} folder
    ############################################################
    clms = [args.x_label, args.y_label]
    ##########################
    dct = {"bace": 0, "bbbp2k": 0, "clintox": -1, "tox21": -1, "lipo": 0, "esol": 0, "freesolv": 0, "bace_reg": 0}   # y tasklarının numarasını belli ediyor!
    ##########################
    dct2 = {"bace": "bace_classification", "bbbp2k": "bbbp", "bace_reg": "bace_regression", "esol": "delaney"}
    ##########################
    if args.ptn in dct2:
        task_name = dct2[args.ptn]
    else:
        task_name = args.ptn
    ############################################################
    attr_name = f"load_{task_name}"
    func = getattr(dc.molnet, attr_name)
    _, (train, valid, test), _ = func(splitter="scaffold", reload=False)
    ############################################################
    train = pd.concat([pd.DataFrame(train.ids), pd.DataFrame(train.y[:,dct[args.ptn]])], axis=1)
    train.columns = clms
    valid = pd.concat([pd.DataFrame(valid.ids), pd.DataFrame(valid.y[:,dct[args.ptn]])], axis=1)
    valid.columns = clms
    test = pd.concat([pd.DataFrame(test.ids), pd.DataFrame(test.y[:,dct[args.ptn]])], axis=1)
    test.columns = clms
    allx = pd.concat([train, valid, test], axis=0)
    ############################################################
    train.to_csv(f"{args.data_folder}/{args.ptn}/{args.ptn}_train.csv")
    valid.to_csv(f"{args.data_folder}/{args.ptn}/{args.ptn}_val.csv")
    test.to_csv(f"{args.data_folder}/{args.ptn}/{args.ptn}_test.csv")
    allx.to_csv(f"{args.data_folder}/{args.ptn}/{args.ptn}_all.csv")
    ############################################################
    return

############################################################

def main(args):
    ####################################################
    data_names = args.data_names.replace(" ", "")
    data_names = data_names.split(",")
    ####################################################
    for ptn in data_names:
        args.ptn = ptn
        get_molnet_from_deepchem(args)
        print(args.ptn)
    ####################################################

############################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--ptn", default="", type=str)   # ptn = pre_task_name
    parser.add_argument("--x_label", default="smiles", type=str)
    parser.add_argument("--y_label", default="y_true", type=str)
    parser.add_argument("--data_folder", default="data", type=str)
    parser.add_argument("--data_names", default="", type=str)
    parser.add_argument("--data_name", default="", type=str)
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


























