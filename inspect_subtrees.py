########################################################################################
########################################################################################

import os
import json
import shutil
from utils.inspection_tools import *

########################################################################################
########################################################################################

def main(data_name, thr1, thr2, contour_num, xt, yt):
    ########################################################################################
    task_path = (f"{args.save_dir}/{data_name}")
    image_path = (f"{task_path}/images")
    ##########################
    # if os.path.exists(task_path):   # klasör varsa, inspection_results/task
    #     shutil.rmtree(task_path)   # klasör siliyor, inspection_results/task
    ##########################
    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir, exist_ok=True)   # klasör oluşturuyor, inspection_results
    # os.mkdir(task_path)   # klasör oluşturuyor, inspection_results/task
    ##########################
    if not os.path.exists(task_path):
        os.mkdir(task_path, exist_ok=True)
        os.mkdir(image_path, exist_ok=True)
    ##########################
    newicks_load_path = (f"{args.load_dir}/{data_name}/all_newicks_{data_name}.json")
    with open(newicks_load_path, "r") as f:
        task_newicks = json.load(f)
    ##########################
    encoder_load_path = f"{args.data_folder}/CHARSET.json"
    with open(encoder_load_path, "r") as f:
        encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    ########################################################################################
    ########################################################################################
    try:
        newicks_load_path = (f"{args.save_dir}/{data_name}/all_subtrees_{data_name}.json")
        with open(newicks_load_path, "r") as f:
            all_subtrees = json.load(f)
    ####################################################
    except:
        all_subtrees, _, _ = find_fragments(task_newicks, decoder)
        ##########################
        newicks_save_path = (f"{args.save_dir}/{data_name}/all_subtrees_{data_name}.json")
        with open(newicks_save_path, "w") as f:
            json.dump(all_subtrees, f)
    ########################################################################################
    ########################################################################################
    try:
        repeat_dict_load_path = (f"{args.save_dir}/{data_name}/repeat_dict_{data_name}.json")
        with open(repeat_dict_load_path, "r") as f:
            repeat_dict = json.load(f)
    ####################################################
    except:
        repeat_dict = inspect_fragments(all_subtrees, task_newicks)
        ##########################
        repeat_dict_save_path = (f"{args.save_dir}/{data_name}/repeat_dict_{data_name}.json")
        with open(repeat_dict_save_path, "w") as f:
            json.dump(repeat_dict, f)
    ########################################################################################
    ########################################################################################
    _ = plot_contour(all_subtrees, repeat_dict, data_name, thr1, thr2, contour_num, xt, yt, "ur")
    _ = plot_contour(all_subtrees, repeat_dict, data_name, thr1, thr2, contour_num, xt, yt, "tr")
    ########################################################################################    
    
########################################################################################    
########################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--data_names", default="", type=str)
    parser.add_argument("--thr", default=20, type=int)
    parser.add_argument("--cbr", default=0.15, type=float)
    parser.add_argument("--data_folder", default="data_new", type=str)
    parser.add_argument("--load_dir", default="../results/evaluation_results", type=str)
    parser.add_argument("--save_dir", default="../results/inspection_results", type=str)
    ##########################
    args = parser.parse_args()
    ##########################    
    return args

########################################################################################    
########################################################################################

if __name__ == "__main__":
    ##########################
    args = load_args()
    ##########################
    cn = int(1 / args.cbr)
    ##########################
    graph_props = {
                   "bace": [args.thr, args.cbr, cn, 50, 4000],
                   "bbbp2k": [args.thr, args.cbr, cn, 50, 200],
                   "clintox": [args.thr, args.cbr, cn, 100, 1000],
                   "tox21": [args.thr, args.cbr, cn, 50, 200],
                   "bbbp8k": [args.thr, args.cbr, cn, 50, 200],
                   "lipo": [args.thr, args.cbr, cn, 50, 200],
                   "esol": [args.thr, args.cbr, cn, 50, 200]
                   }
    ##########################
    data_names = args.data_names.replace(" ", "")
    data_names = data_names.split(",")
    for data_name in data_names:
        lst = graph_props[data_name]
        main(data_name, lst[0], lst[1], lst[2], lst[3], lst[4])
    ##########################

########################################################################################    
########################################################################################















########################################################################################
# from rdkit import Chem
# from rdkit.Chem import Draw
########################################################################################
# smis = ["c1cc(ccc1)", "c1ccccc1", "c1cc(ccc1)C", "Cc1ccccc1"]
# for smi in smis:
    # smi = Chem.CanonSmiles(smi)
#     m = Chem.MolFromSmiles(smi)
#     fig = Draw.MolToMPL(m)   # , size=(1000, 1000)
########################################################################################












































































