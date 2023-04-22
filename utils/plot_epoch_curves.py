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
        lenf = len(filex)
        #######################################
        x_arr = np.linspace(1, lenf, lenf, dtype=int)
        xtcks = np.linspace(0, lenf, 6)
        #######################################
        fig, ax = plt.subplots()
        #########################################################################
        if cv_all[data_name]["task"][0] == "clf":
            plt.plot(x_arr, np.array(filex["bce loss train"]), lw=args.lw, c="blue", label="Training Loss", zorder=1)
            plt.plot(x_arr, np.array(filex["bce loss valid"]), lw=args.lw, c="red", label="Validation Loss", zorder=2)
            plt.plot(x_arr, np.array(filex["roc-auc"]), lw=args.lw, c="green", label="Validation ROC-AUC", zorder=3)
            # plt.plot(x_arr, np.array(filex["prc-auc"]), lw=5, c="black", label="Validation PRC-AUC", zorder=1)
            # plt.plot(x_arr, np.array(filex["accuracy"]), lw=5, c="black", label="Validation Accuracy", zorder=2)
            #######################################
            plt.yticks(np.arange(0.0, 1.1, 0.1))
            #######################################
            plt.ylim(0, 1)
            plt.xlim(1, lenf)
        #########################################################################
        elif cv_all[data_name]["task"][0] == "reg":   # yeni sonuçlar için y = np.sqrt() YAPILACAK !!
            plt.plot(x_arr, np.array(filex["rmse loss train"]), lw=args.lw, c="blue", label="Training RMSE", zorder=1)
            plt.plot(x_arr, np.array(filex["rmse loss valid"]), lw=args.lw, c="red", label="Validation RMSE", zorder=2)
            #######################################
            plt.yticks(np.arange(0.0, 1.6, 0.1))
            #######################################
            plt.ylim(0, 1.5)
            plt.xlim(1, lenf)
        #########################################################################
        plt.title(f"\n{data_name.upper()} Task Epoch Curves\n", fontsize=70)
        #######################################
        plt.ylabel(f"\nScores\n")
        plt.xlabel(f"\nEpoch Number\n")
        #######################################
        plt.xticks(xtcks)
        #######################################
        plt.legend(loc=legend_loc)
        #########################################################################
        save_loc = (f"../results/training_curves/{data_name.upper()}_curves.png")
        plt.savefig(fname=save_loc, bbox_inches='tight')
        #######################################
        plt.close(fig)
        #######################################
        print(f">>  Plotting the curves of {data_name.upper()} task is COMPLETED.\n")
        #########################################################################

#########################################################################
#########################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
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

















































