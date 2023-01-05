########################################################################################
########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################
######################################################################################## IMPORT LIBRARIES

from ete3 import Tree
import json
from rdkit import Chem
from tqdm import tqdm

########################################################################################
######################################################################################## LOAD MAIN TREES

data_name = "bbbp2k"
newicks_load_path = (f"../results/all_newicks_{data_name}.json")
with open(newicks_load_path, "r") as f:
    task_newicks = json.load(f)

########################################################################################
######################################################################################## EXTRACT SUBTREES

all_subtrees = {}
nope_dict = {}
with tqdm(task_newicks.items(), unit="molecule") as tqdm_bar:
    for smi, lst in enumerate(tqdm_bar):
        main_newick = lst[0]   # "(B,(D,(-,F)E)C)A;"
        test_loss = lst[1]
        # if test_loss < 0.5:   # kötülerde de var mı?? acepp
        #     continue
        t = Tree(main_newick, format=8)
        nodes = [node for node in t.traverse()]
        temp_subtrees = []
        for i in range(len(nodes)):
            node = nodes[i]
            if i == 0:
                continue
            if node.is_leaf():
                continue
            point, max_len = get_point(node)
            sub_newick = node.write(format=8)
            # subtree_ascii = node.get_ascii(show_internal=True)   # format=8 ??
            smi = recoverFragment(sub_newick)
            if smi in nope_dict:
                continue
            is_valid = validity_check(smi)
            if not is_valid:
                nope_dict[smi] = [0, 0]
                continue
            all_subtrees[sub_newick] = [point, max_len]

########################## A LITTLE FUNCTION

def get_point(t):
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    nodes = [node for node in t.traverse()]
    total_point = 0
    for i in range(len(nodes)):
        if node.name == "-":
            continue
        point = max_len - t.get_distance(root, node)
        total_point += point
    return total_point, max_len

########################################################################################
######################################################################################## RDKIT VALIDITY CHECK

def validity_check(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        validity = False
    else:
        validity = True
    return validity

########################################################################################    
######################################################################################## SAVE SUBTREES

newicks_save_path = (f"../results/all_{data_name}_subtrees.json")
with open(newicks_save_path, "w") as f:
    json.dump(all_subtrees, f)

########################################################################################
######################################################################################## SEARCH SUBTREES
        
repeat_dict = {}
with tqdm(task_newicks.items(), unit="molecule") as tqdm_bar:
    for smi, main_newick in enumerate(tqdm_bar):
        for sub_newick in all_subtrees:
            point = all_subtrees[sub_newick][0]
            cnt = 0
            repeat_dict = check_subtrees(sub_newick, main_newick, repeat_dict)

########################## A LITTLE FUNCTION
        
def check_subtrees(sub_newick, main_newick, repeat_dict):
    copy_main_newick = main_newick
    while sub_newick in copy_main_newick:
        if sub_newick not in repeat_dict:
            repeat_dict[sub_newick] = [1, 1]
        else:
            temp = repeat_dict[sub_newick][1]
            temp += 1   
            repeat_dict[sub_newick] = [repeat_dict[sub_newick][0], temp]
        copy_main_newick = copy_main_newick.replace(sub_newick, "")
    return repeat_dict

########################################################################################
######################################################################################## SAVE REPEAT DICT

repeat_dict_save_path = (f"../results/repeat_dict_{data_name}.json")
with open(repeat_dict_save_path, "w") as f:
    json.dump(repeat_dict, f)

########################################################################################                
########################################################################################
            
def recoverFragment(sub_newick):
    t = Tree(sub_newick, format=8)
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    print(t_ascii)
    while max_len > 0.0:
        for node in t.traverse():
            if not node.is_leaf():   # if node is not leaf
                continue
            parent = node.up
            if node.name != "-": 
                if node.name == parent.children[0].name:   # means that this node is RIGHT CHILD ! 
                    parent.name = parent.name + parent.children[0].name
                else:   # means that this node is LEFT CHILD !
                    parent.name = parent.children[1].name + parent.name
                node.name = "-"
            if (parent.children[0].name and parent.children[1].name) == "-":
                parent.remove_child(parent.children[1])
                parent.remove_child(parent.children[0])
        t_ascii = t.get_ascii(show_internal=True)
        print(t_ascii)
        _, max_len = root.get_farthest_leaf()
    smi = root.name
    return smi

########################################################################################                
########################################################################################
    









































