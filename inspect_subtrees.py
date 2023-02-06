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
    load_dir = "../results/evaluation_results"
    save_dir = "../results/inspection_results"
    ##########################
    # temp_path = (f"{save_dir}/{data_name}")
    # if os.path.exists(temp_path):   # klasör varsa, inspection_results/task
    #     shutil.rmtree(temp_path)   # klasör siliyor, inspection_results/task
    ##########################
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir, exist_ok=True)   # klasör oluşturuyor, inspection_results
    # os.mkdir(temp_path)   # klasör oluşturuyor, inspection_results/task
    ##########################
    newicks_load_path = (f"{load_dir}/{data_name}/all_newicks_{data_name}.json")
    with open(newicks_load_path, "r") as f:
        task_newicks = json.load(f)
    ##########################
    encoder_load_path = "data/CHARSET.json"
    with open(encoder_load_path, "r") as f:
        encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    ########################################################################################
    ########################################################################################
    try:
        newicks_load_path = (f"{save_dir}/{data_name}/all_subtrees_{data_name}.json")
        with open(newicks_load_path, "r") as f:
            all_subtrees = json.load(f)
    ####################################################
    except:
        all_subtrees, _, _ = find_fragments(task_newicks, decoder)
        ##########################
        newicks_save_path = (f"{save_dir}/{data_name}/all_subtrees_{data_name}.json")
        with open(newicks_save_path, "w") as f:
            json.dump(all_subtrees, f)
    ########################################################################################
    ########################################################################################
    try:
        repeat_dict_load_path = (f"{save_dir}/{data_name}/repeat_dict_{data_name}.json")
        with open(repeat_dict_load_path, "r") as f:
            repeat_dict = json.load(f)
    ####################################################
    except:
        repeat_dict = inspect_fragments(all_subtrees, task_newicks)
        ##########################
        repeat_dict_save_path = (f"{save_dir}/{data_name}/repeat_dict_{data_name}.json")
        with open(repeat_dict_save_path, "w") as f:
            json.dump(repeat_dict, f)
    ########################################################################################
    ########################################################################################
    _ = plot_contour(all_subtrees, repeat_dict, data_name, thr1, thr2, contour_num, xt, yt, "ur")
    _ = plot_contour(all_subtrees, repeat_dict, data_name, thr1, thr2, contour_num, xt, yt, "tr")
    ########################################################################################    
    
########################################################################################    
########################################################################################

if __name__ == "__main__":
    thr = 0.15
    cn = int(1 / thr)
    graph_props = {
                   "bace": [thr, thr, cn, 100, 100],
                   "bbbp2k": [thr, thr, cn, 100, 400],
                   "clintox": [thr, thr, cn, 10, 200],
                   "hiv": [thr, thr, cn, 100, 100],
                   "tox21": [thr, thr, cn, 100, 100]
                   }
    data_name = "tox21"
    lst = graph_props[data_name]
    main(data_name, lst[0], lst[1], lst[2], lst[3], lst[4])

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












































































