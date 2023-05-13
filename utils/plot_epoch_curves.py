#########################################################################
#########################################################################

# cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

#########################################################################
#########################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
import os
import argparse
sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")

#########################################################################
#########################################################################

def main(args):
    #########################################################################
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)   # , exist_ok=True
    #########################################################################
    data_names = os.listdir(args.load_dir)
    #######################################
    with open("cv_config.json", "r") as f:
        cv_all = json.load(f)
    #########################################################################
    print("\n")
    for data_name in data_names:
        # data_name = "bace"
        #######################################
        if data_name == "clintox":
            legend_loc = "center right"
        elif data_name == "tox21":
            if args.pr == "prc":
                legend_loc = "center right"
        else:
            legend_loc = "lower left"
        #######################################
        file_loc = (f"{args.load_dir}/{data_name}/{data_name}_metrics_0.json")
        #######################################
        with open(file_loc, "r") as f:
            file = json.load(f)
        #######################################
        filex = pd.DataFrame(file)
        #######################################
        x_maxx_dct = {"bbbp2k": 200, "sider": 150, "bace_reg": 100, "freesolv": 100}
        if data_name in x_maxx_dct:
            lenf = x_maxx_dct[data_name]
        else:
            lenf = len(filex)
        #######################################
        x_arr = np.linspace(1, lenf, lenf, dtype=int)
        xtcks = np.linspace(0, lenf, 6)
        #######################################
        fig, ax = plt.subplots()
        #########################################################################
        if cv_all[data_name]["task"][0] == "clf":
            if args.pr == "roc":
                plt.plot(x_arr, np.array(filex["bce loss train"][:lenf]), lw=args.lw, c="blue", label="Training Loss", zorder=1)
                plt.plot(x_arr, np.array(filex["bce loss valid"][:lenf]), lw=args.lw, c="red", label="Validation Loss", zorder=2)
                plt.plot(x_arr, np.array(filex["roc-auc"][:lenf]), lw=args.lw, c="green", label="Validation ROC-AUC", zorder=3)
                tag = "roc"
            elif args.pr == "prc":
                plt.plot(x_arr, np.array(filex["prc-auc"][:lenf]), lw=args.lw, c="blue", label="Validation PRC-AUC", zorder=1)
                plt.plot(x_arr, np.array(filex["accuracy"][:lenf]), lw=args.lw, c="green", label="Validation Accuracy", zorder=2)
                tag = "prc"
            #######################################
            plt.yticks(np.arange(0.0, 1.1, 0.1))
            #######################################
            plt.ylim(0, 1)
            plt.xlim(1, lenf)
        #########################################################################
        elif cv_all[data_name]["task"][0] == "reg":   # yeni sonuçlar için y = np.sqrt() YAPILACAK !!
            plt.plot(x_arr, np.array(filex["rmse loss train"][:lenf]), lw=args.lw, c="blue", label="Training RMSE", zorder=1)
            plt.plot(x_arr, np.array(filex["rmse loss valid"][:lenf]), lw=args.lw, c="red", label="Validation RMSE", zorder=2)
            tag = "reg"
            #######################################
            plt.yticks(np.arange(0.0, 1.6, 0.1))
            #######################################
            plt.ylim(0, 1.5)
            plt.xlim(1, lenf)
        #########################################################################
        data_namex = deal_with_names(data_name)
        plt.title(f"\n{data_namex} Task\nScores vs Epoch Curves\n", fontsize=70)
        #######################################
        plt.ylabel(f"\nScores\n")
        plt.xlabel(f"\nEpoch Number\n")
        #######################################
        plt.xticks(xtcks)
        #######################################
        plt.legend(loc=legend_loc)
        #########################################################################
        save_loc = (f"../results/training_curves/{data_name.upper()}_{tag.upper()}_curves.png")
        plt.savefig(fname=save_loc, bbox_inches='tight')
        #######################################
        plt.close(fig)
        #######################################
        print(f">>  Plotting the curves of {data_name.upper()} task is COMPLETED.\n")
        #########################################################################

#########################################################################
#########################################################################

def deal_with_names(n):
    tmp_dct = {
               "bace": "BACE Classification",
               "bace_reg": "BACE Regression",
               "bbbp2k": "BBBP-2K",
               "bbbp8k": "BBBP-10K",
               "clintox": "ClinTox",
               "freesolv": "FreeSolv",
               "lipo": "Lipophilicity",
               "tox21": "Tox21"
               }
    if n in tmp_dct:
        n = tmp_dct[n]
    else:
        n = n.upper()
    return n

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--pr", default="roc", choices=["roc", "prc"], type=str)
    parser.add_argument("--load_dir", default="../results/training_results", type=str)
    parser.add_argument("--save_dir", default="../results/training_curves", type=str)
    parser.add_argument("--lw", default=5, type=int)
    ##########################
    args = parser.parse_args()
    ##########################    
    return args

#########################################################################
#########################################################################

if __name__ == "__main__":
    ##########################
    args = load_args()
    main(args)
    ##########################

#########################################################################
#########################################################################

















































