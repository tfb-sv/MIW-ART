########################################################################################
########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################
########################################################################################

from tqdm import tqdm
import json
from utils.inspection_tools_v2 import *

########################################################################################
########################################################################################

newicks_load_path = "../results/evaluation_results/clintox/all_newicks_clintox.json"
with open(newicks_load_path, "r") as f:
    task_newicks = json.load(f)

####################################################

encoder_load_path = "data/CHARSET.json"
with open(encoder_load_path, "r") as f:
    encoder = json.load(f)
decoder = {v: k for k, v in encoder.items()}

########################################################################################

allData = []
# cnt = 0
with tqdm(task_newicks.items(), unit=" molecule", disable=True) as tqdm_bar:
    for nwck_cnt, lst in enumerate(tqdm_bar):
        orj_smi = lst[0]
        main_newick = lst[1][0]
        test_loss = lst[1][1]
        y_label = lst[1][2]
        if y_label == 0:
            continue
        decoded_lst, is_ok = recoverTree(main_newick, decoder, orj_smi)   # 1478 tane de 111 tanesi toparlanamadı bir türlü ne yazık ki...
        # if not is_ok:
            # cnt += 1
            # print(cnt)
            # continue
        point_lst = recoverPoints(main_newick, decoder, orj_smi)
        allData.append([orj_smi, decoded_lst, point_lst])
        # if len(decoded_lst) == len(point_lst):
        #     allData.append([orj_smi, decoded_lst, point_lst])
        
        
########################################################################################
########################################################################################











































































