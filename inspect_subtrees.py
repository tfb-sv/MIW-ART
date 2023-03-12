########################################################################################
########################################################################################

import os
import json
import shutil
from utils.inspection_tools import *

########################################################################################
########################################################################################

def main(args):
    ########################################################################################
    task_path = (f"{args.save_dir}/{args.data_name}")
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
        os.mkdir(task_path)   # , exist_ok=True
    if not os.path.exists(image_path):
        os.mkdir(image_path)   # , exist_ok=True
    ##########################
    newicks_load_path = (f"{args.load_dir}/{args.data_name}/all_newicks_{args.data_name}.json")
    with open(newicks_load_path, "r") as f:
        task_newicks = json.load(f)
    ##########################
    encoder_load_path = f"{args.data_folder}/CHARSET.json"
    with open(encoder_load_path, "r") as f:
        encoder = json.load(f)
    decoder = {v: k for k, v in encoder.items()}
    ######################################################################################## 
    ########################################################################################    
    total_loss = 0
    total_cnt = 0
    for smi in task_newicks:
        test_loss = task_newicks[smi][1]
        test_label = task_newicks[smi][2]
        total_loss += np.square(test_loss - test_label)
        total_cnt += 1
    task_avg_loss = np.round((total_loss / total_cnt), 4)
    args.task_avg_loss = task_avg_loss
    ########################################################################################
    ########################################################################################
    try:
        newicks_load_path = (f"{args.save_dir}/{args.data_name}/all_subtrees_{args.data_name}.json")
        with open(newicks_load_path, "r") as f:
            all_subtrees = json.load(f)
    ####################################################
    except:
        all_subtrees, _, _ = find_fragments(task_newicks, decoder)
        ##########################
        newicks_save_path = (f"{args.save_dir}/{args.data_name}/all_subtrees_{args.data_name}.json")
        with open(newicks_save_path, "w") as f:
            json.dump(all_subtrees, f)
    ########################################################################################
    ########################################################################################
    try:
        repeat_dict_load_path = (f"{args.save_dir}/{args.data_name}/repeat_dict_{args.data_name}.json")
        with open(repeat_dict_load_path, "r") as f:
            repeat_dict = json.load(f)
    ####################################################
    except:
        repeat_dict = inspect_fragments(all_subtrees, task_newicks, args.task_avg_loss, args.task)
        ##########################
        repeat_dict_save_path = (f"{args.save_dir}/{args.data_name}/repeat_dict_{args.data_name}.json")
        with open(repeat_dict_save_path, "w") as f:
            json.dump(repeat_dict, f)
    ########################################################################################
    ########################################################################################
    # _ = plot_contour(all_subtrees, repeat_dict, args.data_name, | thr1, thr2, contour_num, xt, yt, "ur", task, | task_avg_loss)
    _ = plot_contour(all_subtrees, repeat_dict, args)
    ########################################################################################   
    ########################################################################################
    
########################################################################################    
########################################################################################

def load_args():
    ##########################
    parser = argparse.ArgumentParser()
    ##########################
    parser.add_argument("--data_names", default="", type=str)
    parser.add_argument("--data_folder", default="data", type=str)
    parser.add_argument("--load_dir", default="../results/evaluation_results", type=str)
    parser.add_argument("--save_dir", default="../results/inspection_results", type=str)
    ####################################################
    parser.add_argument("--task", default="clf", type=str)
    parser.add_argument("--thr", default=20, type=int)
    parser.add_argument("--cbr", default=0.25, type=float)
    parser.add_argument("--contour_level_line", default=0, type=int)
    parser.add_argument("--xt", default=2, type=int)
    parser.add_argument("--yt", default=2, type=int)
    parser.add_argument("--repeat_type", default="ur", choices=["ur", "tr"], type=str)
    parser.add_argument("--data_name", default="", type=str)
    parser.add_argument("--task_avg_loss", default=0.0, type=float)
    parser.add_argument("--contour_level_fill", default=250, type=int)
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
    args.contour_level_line = int(1 / args.cbr)
    ##########################
    main_dir = os.getcwd()
    with open(f"{main_dir}/cv_config.json", "r") as f:
        cv_config = json.load(f)
    ##########################
    data_names = args.data_names.replace(" ", "")
    data_names = data_names.split(",")
    for data_name in data_names:
        args.data_name = data_name
        args.task = cv_config[args.data_name]["task"]
        main(args)
    ##########################

########################################################################################    
########################################################################################




# data_name, thr, cbr, cn                , xt,   yt,   repeat_type, task, task_avg_loss   # for ARGs >
# data_name, thr1,thr2,contour_num,        xt,   yt,   "ur",        task, task_avg_loss   # for MAIN FUNC >
# data_name, thr, cbr, contour_level_line, atrx, atry, repeat_type, task, task_avg_loss   # for plot_contour FUNC
    
    













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












































































