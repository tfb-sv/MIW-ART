########################################################################################
########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################

import json
# from rdkit import Chem
# from rdkit.Chem import Draw
from utils.inspection_tools import *

########################################################################################
########################################################################################

data_name = "bace"

##########################

# newicks_load_path = (f"../results/all_newicks_{data_name}.json")
newicks_load_path = (f"data/all_encoded_newicks_{data_name}.json")
with open(newicks_load_path, "r") as f:
    task_newicks = json.load(f)

##########################

# decoder_load_path = "data/INV_CHARSET.json"
# with open(decoder_load_path, "r") as f:
#     decoder = json.load(f)

########################################################################################
########################################################################################

# all_subtrees, _, _ = find_fragments(task_newicks, decoder)

##########################

# newicks_save_path = (f"../results/all_{data_name}_subtrees.json")
# with open(newicks_save_path, "w") as f:
#     json.dump(all_subtrees, f)

##########################

newicks_load_path = (f"../results/all_{data_name}_subtrees.json")
with open(newicks_load_path, "r") as f:
    all_subtrees = json.load(f)

########################################################################################

# repeat_dict = inspect_fragments(all_subtrees, task_newicks)

##########################

# repeat_dict_save_path = (f"../results/repeat_dict_{data_name}.json")
# with open(repeat_dict_save_path, "w") as f:
#     json.dump(repeat_dict, f)

##########################

repeat_dict_load_path = (f"../results/repeat_dict_{data_name}.json")
with open(repeat_dict_load_path, "r") as f:
    repeat_dict = json.load(f)

########################################################################################
########################################################################################

_ = plot_contour(all_subtrees, repeat_dict, data_name, 4000, 4000, 100, "ur")
    
########################################################################################    
########################################################################################

# smis = ["c1cc(ccc1)", "c1ccccc1", "c1cc(ccc1)C", "Cc1ccccc1"]
# for smi in smis:
    # smi = Chem.CanonSmiles(smi)
#     m = Chem.MolFromSmiles(smi)
#     fig = Draw.MolToMPL(m)   # , size=(1000, 1000)
    
########################################################################################
########################################################################################



















































