########################################################################################
########################################################################################

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
# from scipy.interpolate import griddata
RDLogger.DisableLog('rdApp.*')
# sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")

########################################################################################
########################################################################################

def decode_newick(enc_smis, decoder, smi, node_cnt):
    enc_smis = enc_smis.split("a")
    decoded_lst = [decoder[e] for e in enc_smis]
    if len(decoded_lst) != node_cnt:
        print("\n", decoded_lst, "\n", len(decoded_lst), node_cnt)
        dslşkfsdf
    decoded_lst.reverse()
    sub_smi = "".join(decoded_lst)   # decoded = ...
    # decoded = decoded.replace("{", "(")
    # sub_smi = decoded.replace("}", ")")
    is_ok = sanity_check(smi, sub_smi)
    if not is_ok:
        inv_sub_smi = sub_smi[::-1]
        # print(f"\n>>  {smi}\n>>  {sub_smi}\n>>  {inv_sub_smi}")
        # dfşlsşdf
    if len(decoded_lst) < 3:
        is_ok = False
    return is_ok, sub_smi

########################################################################################

def get_point(t):
    nodes = [node for node in t.traverse()]
    root = nodes[0]   # t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    total_point = 0
    for i in range(len(nodes)):
        node = nodes[i]
        if node.name == "-":
            continue
        point = (max_len + 1) - t.get_distance(root, node)
        total_point += point
    return total_point, max_len

########################################################################################

def validity_check(smi):
    m = Chem.MolFromSmiles(smi, sanitize=True)
    if m is None:
        validity = False
    else:
        validity = True
    return validity

########################################################################################

def search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict):
    is_unique = True
    sub_csmi = Chem.CanonSmiles(sub_smi)
    while sub_smi in smi:
        point = frag_size * fixed_loss
        if sub_csmi not in repeat_dict:
            repeat_dict[sub_csmi] = [1, 1, point]
        else:
            temp1 = repeat_dict[sub_csmi][0]
            if is_unique:
                temp1 += 1
            temp2 = repeat_dict[sub_csmi][1]
            temp2 += 1   
            temp3 = repeat_dict[sub_csmi][2]
            temp3 += point
            repeat_dict[sub_csmi] = [temp1, temp2, temp3]
        smi = smi.replace(sub_smi, "", 1)
        is_unique = False
    return repeat_dict

########################################################################################

def recoverFragment(sub_newick, decoder, smi):
    ##########################
    t = Tree(sub_newick, format=8)
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    cnt = 0
    for node in t.traverse():
        if node.name != "-":
            cnt += 1
    # t_ascii = t.get_ascii(show_internal=True)
    # print(t_ascii)
    ##########################
    while max_len > 0.0:
        for node in t.traverse():
            if not node.is_leaf():   # if node is not leaf, just pass this node.
                continue
            parent = node.up
            if node.name == "-":
                continue
            ##########################
            child_right = parent.children[0]
            child_left = parent.children[1]
            if node.name == child_right.name:   # means that this node is RIGHT CHILD ! 
                parent.name = parent.name + "a" + child_right.name
            else:   # means that this node is LEFT CHILD !
                parent.name = child_left.name + "a" + parent.name
            ##########################
            node.name = "-"
            if child_right.name == "-":
                if child_left.name == "-":
                    parent.remove_child(child_left)
                    parent.remove_child(child_right)
        ##########################
        # t_ascii = t.get_ascii(show_internal=True)
        # print(t_ascii)
        _, max_len = root.get_farthest_leaf()
    is_ok, sub_smi = decode_newick(root.name, decoder, smi, cnt)
    ##########################
    return is_ok, sub_smi

########################################################################################

def sanity_check(smi, sub_smi):
    is_ok = True
    if sub_smi not in smi:
        is_ok = False
    return is_ok

########################################################################################

def find_fragments(task_newicks, decoder):
    all_subtrees, not_valid_dict, not_ok_dict = {}, {}, {}
    with tqdm(task_newicks.items(), unit="molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]   # "(((3,-)4,(1,-)11)11,-)8;"   #    # "(B,(D,(-,F)E)C)A;"   # 
            test_loss = lst[1][1]
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
                is_ok, sub_smi = recoverFragment(sub_newick, decoder, smi)
                if sub_smi in not_valid_dict:
                    continue
                is_valid = validity_check(sub_smi)
                if not is_valid:
                    not_valid_dict[sub_smi] = ["0"]
                    continue
                if is_ok:
                    all_subtrees[sub_newick] = [sub_smi, point, max_len]
                else:
                    not_ok_dict[sub_smi] = [smi]
    return all_subtrees, not_valid_dict, not_ok_dict

########################################################################################

def fix_loss(loss):
    fixed_loss = np.round((1 - np.tanh(loss)), 4)
    return fixed_loss

########################################################################################

def passFilter(x, sign, thr):
    return x

########################################################################################

def inspect_fragments(all_subtrees, task_newicks):
    ##########################
    repeat_dict = {}
    with tqdm(task_newicks.items(), unit="molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]
            test_loss = lst[1][1]
            fixed_loss = fix_loss(test_loss)
            ##########################
            for sub_newick in all_subtrees:
                sub_smi = all_subtrees[sub_newick][0]
                frag_size = all_subtrees[sub_newick][1]
                # max_len = all_subtrees[sub_newick][2]
                repeat_dict = search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict)    
    ##########################
    return repeat_dict

########################################################################################

def set_xyz(all_subtrees, repeat_dict):
    ##########################
    frag_size_dict = {}
    for key in all_subtrees:
        smi = all_subtrees[key][0]
        smi = Chem.CanonSmiles(smi)
        frag_size = all_subtrees[key][1]
        frag_size_dict[smi] = frag_size
    ##########################
    xyz = {}
    x_lst, y_lst, z_lst, smis, w_lst = [], [], [], [], []
    # cids, names = [], []
    ##########################
    for smi in repeat_dict:
        # compound = pcp.get_compounds(smi, "smiles")[0]
        # cid = compound.cid
        # name = compound.name
        unique_repeat = repeat_dict[smi][0]
        total_repeat = repeat_dict[smi][1]
        total_point = repeat_dict[smi][2]
        total_point = int(np.round(total_point, 0))
        ##########################
        frag_size = frag_size_dict[smi]
        frag_size = int(frag_size)
        ##########################
        if unique_repeat < 2:
            continue
        ##########################
        x_lst.append(frag_size) 
        y_lst.append(unique_repeat)
        z_lst.append(total_point)
        w_lst.append(total_repeat)
        smis.append(smi)
        # cids.append(cid)
        # names.append(name)
    ##########################
    xyz["x"] = x_lst
    xyz["y"] = y_lst
    xyz["z"] = z_lst
    xyz["w"] = w_lst
    xyz["smi"] = smis
    # xyz["cid"] = cids
    # xyz["name"] = names
    ##########################
    return xyz, frag_size_dict

######################################################################################## 

def plot_contour(all_subtrees, repeat_dict, data_name, thr, cbr, atr, repeat_type):
    ####################################################
    xyz, _ = set_xyz(all_subtrees, repeat_dict)
    ##########################
    ur = xyz["y"]
    tr = xyz["w"]
    ##########################
    x = xyz["x"]
    if repeat_type == "ur":
        y = ur
        y_label = "\nUnique Repeat Count\n"
    elif repeat_type == "tr":
        y = tr
        y_label = "\nTotal Repeat Count\n"
    z = xyz["z"]
    smis = xyz["smi"]
    # cids = xyz["cid"]
    # names = xyz["name"]
    ####################################################
    contour_level_fill = 250
    contour_level_line = 5
    ##########################
    x_ticks = np.arange(0, max(x), atr).tolist()
    x_ticks.append(max(x))
    # x_ticks.append(50)
    # x_ticks = sorted(x_ticks)
    ##########################
    cb_tick_lst = np.arange(0, max(z), cbr).tolist()
    cb_tick_lst.append(max(z))
    ##########################
    y_ticks = np.arange(0, max(y), atr).tolist()
    y_ticks.append(max(y))
    ####################################################
    sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="white")
    x_good, x_bad, y_good, y_bad = [], [], [], []
    cids = []
    for i in range(len(z)):
        if z[i] >= thr:
            x_good.append(x[i])
            y_good.append(y[i])
            c = pcp.get_compounds(smis[i], "smiles")[0]
            cid_str = (f"CID {c.cid}")
            cids.append([cid_str, x[i], y[i]])
            m = Chem.MolFromSmiles(smis[i])
            fig = Draw.MolToMPL(m, size=(200, 200))
            title = (f"{c.iupac_name}\n{cid_str}\n{smis[i]}\n\nUR = {ur[i]}    TR = {tr[i]}\nFS = {x[i]}     TP = {z[i]}")
            fig.suptitle(title, fontsize=35, x=1.25, y=0.7)
            fig.set_size_inches(5, 5)
            save_loc = (f"../results/z.ex/rez/{data_name}")
            if not os.path.exists(save_loc):
                os.mkdir(save_loc)
            save_name = (f"{save_loc}/{repeat_type.upper()} {cid_str}.png")
            plt.savefig(fname=save_name, bbox_inches="tight", dpi=100)
            plt.close(fig)
            print(cid_str)
        else:
            x_bad.append(x[i])
            y_bad.append(y[i])
    ####################################################
    sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")
    fig, ax = plt.subplots()
    plt.tricontourf(x, y, z, contour_level_fill, cmap="CMRmap_r", zorder=1)   # "RdBu_r", "Spectral_r"
    plt.colorbar(ticks=cb_tick_lst, label="\nImportance Metric\n")
    scatter_good = ax.scatter(x_bad, y_bad, marker="o", c="white", zorder=2)
    contours = plt.tricontour(x, y, z, contour_level_line, linewidths=2, colors="white", zorder=3)
    # ax.axvline(50, ymax=max(y), linewidth=3.5, c="black", zorder=4)
    ax.clabel(contours, inline=True, fontsize=20, zorder=5)
    ax.scatter(x_good, y_good, marker="X", s=300, c="white", edgecolors="black", zorder=6)
    # ax.scatter(x_good, y_good, marker="o", s=1500, linewidth=5, facecolors="none", edgecolors="black", zorder=7)
    for i in range(len(cids)):
        ax.annotate(cids[i][0], (cids[i][1], cids[i][2]), zorder=7)
    ax.legend(
              # *scatter_good.legend_elements(),
              ("X", "o"),
              # ("Goods", "Bads"),
              loc="upper right",
              fontsize=50,
              )
    # ax.add_artist(legend)
    ####################################################
    plt.title(f"\nFragment Importance Contour Plot\n({data_name.upper()} Task)\n")   # , fontsize = ?
    plt.ylabel(y_label)   # , fontsize = ?
    plt.xlabel("\nFragment Size\n")   # , fontsize = ?
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.ylim(0, max(y))
    plt.xlim(0, max(x))
    ####################################################
    save_name = (f"{save_loc}/{repeat_type} importance map.png")
    plt.savefig(fname=save_name, bbox_inches='tight')
    plt.close(fig)
    ####################################################
    return xyz

########################################################################################

def visualize_subtrees(all_subtrees, decoder):
    print("\n")
    for sub_newick in all_subtrees:
        t = Tree(sub_newick, format=8)
        for node in t.traverse():
            new_name = decoder[node.name]
            new_name = new_name.replace("(", "{")
            new_name = new_name.replace(")", "}")
            node.name = new_name
        t_ascii = t.get_ascii(format=8)
        print(t_ascii, "\n")
    return

########################################################################################






























































