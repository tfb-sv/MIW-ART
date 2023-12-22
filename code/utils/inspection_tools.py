from ete3 import Tree
import json
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pubchempy as pcp
import os
import re
import argparse
import pandas as pd
RDLogger.DisableLog('rdApp.*')

def check_smiles_length(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    smi_new = "".join(tokens)
    if smi != smi_new: print(f"\n{smi}\n{smi_new}")
    assert smi == smi_new
    if len(tokens) < 3: return False
    else: return True

def decode_newick(enc_smis, decoder, smi, node_cnt):
    enc_smis = enc_smis.split("a")
    decoded_lst = [decoder[int(e)] for e in enc_smis]
    if len(decoded_lst) != node_cnt:
        print("\n", decoded_lst, "\n", len(decoded_lst), node_cnt)
        dslşkfsdf
    decoded_lst.reverse()
    sub_smi = "".join(decoded_lst)
    is_ok = sanity_check(smi, sub_smi)
    if len(decoded_lst) < 3: is_ok = False
    return is_ok, sub_smi

def validity_check(smi):
    m = Chem.MolFromSmiles(smi, sanitize=True)
    if m is None: validity = False
    else: validity = True
    return validity

def search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict, test_loss, label, args):
    sub_csmi = Chem.CanonSmiles(sub_smi)
    is_smi_len_ok = check_smiles_length(sub_csmi)
    if test_loss > (args.thr): return repeat_dict
    else: pass
    if args.task == "reg":
        if label < args.thr2: return repeat_dict
    else:
        if int(label) != 1: return repeat_dict
    if not is_smi_len_ok: return repeat_dict
    else:
        if sub_smi in smi:
            point = frag_size * fixed_loss
            if sub_csmi not in repeat_dict:
                temp_smi_dct = {}
                temp_smi_dct[smi] = test_loss
                repeat_dict[sub_csmi] = [1, 0, point, frag_size, temp_smi_dct]
            else:  
                temp_smi_dct = repeat_dict[sub_csmi][4]   # smis and losses
                if smi not in temp_smi_dct:                  
                    temp_smi_dct[smi] = test_loss
                    repeat_dict[sub_csmi][4] = temp_smi_dct
                    temp0 = repeat_dict[sub_csmi][0]   # unique_repeat
                    temp0 += 1
                    repeat_dict[sub_csmi][0] = temp0
                    temp2 = repeat_dict[sub_csmi][2]   # total_point
                    temp2 += point
                    repeat_dict[sub_csmi][2] = temp2
                    temp3 = repeat_dict[sub_csmi][3]   # total_fragment_size
                    temp3 += frag_size
                    repeat_dict[sub_csmi][3] = temp3
        return repeat_dict

def recoverFragment(sub_newick, decoder, smi):
    t = Tree(sub_newick, format=8)
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    cnt = 0
    for node in t.traverse():
        if node.name != "-": cnt += 1
    while max_len > 0.0:
        for node in t.traverse():
            if not node.is_leaf(): continue   # if node is not leaf, just pass this node
            parent = node.up
            if node.name == "-": continue
            child_right = parent.children[0]
            child_left = parent.children[1]
            if node.name == child_right.name: parent.name = parent.name + "a" + child_right.name   # means that this node is RIGHT CHILD
            else: parent.name = child_left.name + "a" + parent.name   # means that this node is LEFT CHILD
            node.name = "-"
            if child_right.name == "-":
                if child_left.name == "-":
                    parent.remove_child(child_left)
                    parent.remove_child(child_right)
        _, max_len = root.get_farthest_leaf()
    is_ok, sub_smi = decode_newick(root.name, decoder, smi, cnt)
    return is_ok, sub_smi

def sanity_check(smi, sub_smi):
    is_ok = True
    if sub_smi not in smi: is_ok = False
    return is_ok

def find_fragments(task_newicks, decoder, data_name):
    all_subtrees, not_valid_dict, not_ok_dict = {}, {}, {}
    print(f"\n>>  Finding {data_name.upper()} fragments...  <<\n")
    with tqdm(task_newicks.items(), unit=" molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]
            test_loss = lst[1][1]
            y_label = lst[1][2]
            if y_label == 0: continue
            t = Tree(main_newick, format=8)
            nodes = [node for node in t.traverse()]
            temp_subtrees = []
            for i in range(len(nodes)):
                node = nodes[i]
                if i == 0: continue
                if node.is_leaf(): continue
                point, max_len = get_point(node)
                sub_newick = node.write(format=8)
                is_ok, sub_smi = recoverFragment(sub_newick, decoder, smi)
                if sub_smi in not_valid_dict: continue
                is_valid = validity_check(sub_smi)
                if not is_valid:
                    not_valid_dict[sub_smi] = ["0"]
                    continue
                if is_ok: all_subtrees[sub_newick] = [sub_smi, point, max_len]
                else: not_ok_dict[sub_smi] = [smi]
    return all_subtrees, not_valid_dict, not_ok_dict

def get_point(t):
    nodes = [node for node in t.traverse()]
    root = nodes[0]
    _, max_len = root.get_farthest_leaf()
    total_point = 0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.name == "-": continue
        point = (max_len + 1) - t.get_distance(root, node)
        total_point += point
    return total_point, max_len

def fix_loss(loss):
    return np.round((1 - np.tanh(loss)), 4)

def inspect_fragments(all_subtrees, task_newicks, task_avg_loss, task, data_name, args):
    repeat_dict = {}
    print(f"\n>>  Inspecting {data_name.upper()} fragments...  <<\n")
    with tqdm(task_newicks.items(), unit=" molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]
            test_loss = lst[1][1]
            label = lst[1][2]
            fixed_loss = fix_loss(test_loss)
            for sub_newick in all_subtrees:
                sub_smi = all_subtrees[sub_newick][0]
                frag_size = all_subtrees[sub_newick][1]
                repeat_dict = search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict, test_loss, label, args)    
    return repeat_dict

def plot_contour(all_subtrees, repeat_dict, args):   
    xyz, _ = set_xyz(all_subtrees, repeat_dict)
    x = xyz["x"]   # fragment sizes
    y = xyz["y"]   # unique_repeat
    z = xyz["z"]   # total_point
    sub_smis = xyz["sub_smis"]
    smis = xyz["smis"]   # parent smiles and test loss dict
    ur = y
    y_label = "\nloge(Unique Repeat Count)\n"
    x = np.asarray(x)
    x = np.log(x).tolist()
    y = np.asarray(y)
    y = np.log(y).tolist()
    z = np.asarray(z)
    z = z / np.max(z)
    z = z.tolist()
    thr2 = args.task_avg_loss / np.max(z)
    z, x, y, sub_smis, ur = zip(*reversed(sorted(zip(z, x, y, sub_smis, ur))))
    sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="white")
    x_good, y_good = [], []
    z_good = {}
    cids = []
    save_loc = (f"{args.save_dir}/{args.data_name}")
    print("\n")
    cnt = 0
    mols = []
    mcidz = []
    print(f">> {args.data_name.upper()} fragments are being visualized..\n")
    for i in range(len(z)):
        zi = np.round(z[i], 4)
        cnt += 1
        if cnt <= args.thr:
            x_good.append(x[i])
            y_good.append(y[i])
            sub_smi = sub_smis[i]
            z_good[sub_smi] = i
            rank = z[i]
            c = pcp.get_compounds(sub_smi, "smiles")[0]
            cid_str = (f"CID {c.cid}")
            cid_str2 = (f"{(i + 1)} - {cid_str} ({zi:.4f})")
            cid_str_annot = (f"{cid_str} ({zi:.4f})")
            cids.append([cid_str_annot, x[i], y[i]])
            iupac_name = c.iupac_name
            m = Chem.MolFromSmiles(sub_smi)
            mols.append(m)
            mcidz.append(cid_str2)
            parents = pd.DataFrame(list(smis[sub_smi].keys()))
            parents.columns = ["smiles"]
            parents.to_csv(f"{save_loc}/csvs/{(i + 1)} - CID {c.cid}.csv")
            fig = Draw.MolToMPL(m, size=(350, 350))
            title = (f"{iupac_name}\n{cid_str}\n{sub_smi}\n\nUR = {ur[i]}    FS = {x[i]:.2f}     TP = {zi:.4f}\nTask = {args.data_name.upper()}     Rank = {(i + 1)}")
            fig.suptitle(title, fontsize=35, x=1.25, y=0.8)
            fig.set_size_inches(5.5, 5.5)
            if not os.path.exists(save_loc): os.mkdir(save_loc)
            save_name = (f"{save_loc}/images/{(i + 1)} - {cid_str}.png")
            plt.savefig(fname=save_name, bbox_inches="tight", dpi=100)
            plt.close(fig)
            cid_out = f"{cnt} - {cid_str_annot} => {sub_smi}"
            print(cid_out) 
        else: pass
    fig_grid = Draw.MolsToGridImage(mols, 
                                    legends=mcidz, 
                                    molsPerRow=4,
                                    subImgSize=(400, 400))
    save_name = (f"{save_loc}/images/{args.data_name.upper()} All Fragments.png")
    fig_grid.save(save_name, format="PNG")
    return xyz

def set_xyz(all_subtrees, repeat_dict):
    frag_size_dict = {}
    for key in all_subtrees:
        sub_smi = all_subtrees[key][0]
        sub_smi = Chem.CanonSmiles(sub_smi)
        frag_size = all_subtrees[key][1]
        frag_size_dict[sub_smi] = frag_size
    xyz = {}
    x_lst, y_lst, z_lst, sub_smis, w_lst, subs2smis_dct = [], [], [], [], [], {}
    for sub_smi in repeat_dict:
        unique_repeat = repeat_dict[sub_smi][0]
        total_point = repeat_dict[sub_smi][2]
        total_frag_size = repeat_dict[sub_smi][3]
        smis_and_losses = repeat_dict[sub_smi][4]
        total_point = int(np.round(total_point, 0))
        frag_size = np.round((total_frag_size / unique_repeat), 2)
        if unique_repeat < 2: continue
        x_lst.append(frag_size) 
        y_lst.append(unique_repeat)
        z_lst.append(total_point)
        sub_smis.append(sub_smi)
        subs2smis_dct[sub_smi] = smis_and_losses
    xyz["x"] = x_lst
    xyz["y"] = y_lst
    xyz["z"] = z_lst
    xyz["sub_smis"] = sub_smis
    xyz["smis"] = subs2smis_dct
    return xyz, frag_size_dict
    