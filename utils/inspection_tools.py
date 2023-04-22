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
import re
import argparse
import pandas as pd
# from scipy.interpolate import griddata
RDLogger.DisableLog('rdApp.*')

########################################################################################
########################################################################################

def check_smiles_length(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    smi_new = "".join(tokens)
    if smi != smi_new:
        print(f"\n{smi}\n{smi_new}")
    assert smi == smi_new
    if len(tokens) < 3:
        return False
    else:
        return True

def decode_newick(enc_smis, decoder, smi, node_cnt):
    enc_smis = enc_smis.split("a")
    decoded_lst = [decoder[int(e)] for e in enc_smis]
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

def search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict, test_loss, label, args):
    ####################################################
    # is_unique = True
    sub_csmi = Chem.CanonSmiles(sub_smi)
    is_smi_len_ok = check_smiles_length(sub_csmi)
    ####################################################
    if test_loss > (args.thr):
        return repeat_dict
        ##########################
    else:
        pass
    ####################################################
    if args.task == "reg":
        if label < args.thr2:   # kesin yüksek olmasını mı istiyoruz, datasetlere bak !
            return repeat_dict
        # pass
        ##########################
    else:   # "clf"
        if int(label) != 1:
            return repeat_dict
    ####################################################
    if not is_smi_len_ok:
        return repeat_dict
        ##########################
    else:
        if sub_smi in smi:
            point = frag_size * fixed_loss
            #########################
            if sub_csmi not in repeat_dict:
                temp_smi_dct = {}
                temp_smi_dct[smi] = test_loss
                repeat_dict[sub_csmi] = [1, 0, point, frag_size, temp_smi_dct]   # index 1 (2nd one) currently useless now.
                #########################
            else:  
                temp_smi_dct = repeat_dict[sub_csmi][4]   # smis and losses
                ########################
                if smi not in temp_smi_dct:                  
                    temp_smi_dct[smi] = test_loss
                    repeat_dict[sub_csmi][4] = temp_smi_dct
                    ########################
                    temp0 = repeat_dict[sub_csmi][0]   # unique_repeat
                    temp0 += 1
                    repeat_dict[sub_csmi][0] = temp0
                    ########################
                    temp2 = repeat_dict[sub_csmi][2]   # total_point
                    temp2 += point
                    repeat_dict[sub_csmi][2] = temp2
                    ########################
                    temp3 = repeat_dict[sub_csmi][3]   # total_fragment_size
                    temp3 += frag_size
                    repeat_dict[sub_csmi][3] = temp3
                ########################
            #########################
        return repeat_dict
    ####################################################

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

def find_fragments(task_newicks, decoder, data_name):
    all_subtrees, not_valid_dict, not_ok_dict = {}, {}, {}
    print(f"\n>>  Finding {data_name.upper()} fragments...  <<\n")
    with tqdm(task_newicks.items(), unit=" molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]   # "(((3,-)4,(1,-)11)11,-)8;"   #    # "(B,(D,(-,F)E)C)A;"   # 
            test_loss = lst[1][1]
            y_label = lst[1][2]
            if y_label == 0:
                continue
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

def inspect_fragments(all_subtrees, task_newicks, task_avg_loss, task, data_name, args):
    ##########################
    repeat_dict = {}
    print(f"\n>>  Inspecting {data_name.upper()} fragments...  <<\n")
    ####################################################
    with tqdm(task_newicks.items(), unit=" molecule") as tqdm_bar:
        for nwck_cnt, lst in enumerate(tqdm_bar):
            smi = lst[0]
            main_newick = lst[1][0]
            test_loss = lst[1][1]
            label = lst[1][2]
            fixed_loss = fix_loss(test_loss)
            ##########################
            for sub_newick in all_subtrees:
                sub_smi = all_subtrees[sub_newick][0]
                frag_size = all_subtrees[sub_newick][1]
                # max_len = all_subtrees[sub_newick][2]
                repeat_dict = search_subtrees(sub_smi, smi, frag_size, fixed_loss, repeat_dict, test_loss, label, args)    
    ##########################
    return repeat_dict

########################################################################################

def set_xyz(all_subtrees, repeat_dict):
    ##########################
    frag_size_dict = {}
    for key in all_subtrees:
        sub_smi = all_subtrees[key][0]
        sub_smi = Chem.CanonSmiles(sub_smi)
        frag_size = all_subtrees[key][1]
        frag_size_dict[sub_smi] = frag_size
    ##########################
    xyz = {}
    x_lst, y_lst, z_lst, sub_smis, w_lst, subs2smis_dct = [], [], [], [], [], {}
    # cids, names = [], []
    # parents = []
    ##########################
    for sub_smi in repeat_dict:
        # compound = pcp.get_compounds(sub_smi, "smiles")[0]
        # cid = compound.cid
        # name = compound.name
        unique_repeat = repeat_dict[sub_smi][0]
        # total_repeat = repeat_dict[sub_smi][1]
        total_point = repeat_dict[sub_smi][2]
        total_frag_size = repeat_dict[sub_smi][3]
        smis_and_losses = repeat_dict[sub_smi][4]
        ##########################
        total_point = int(np.round(total_point, 0))
        # frag_size = np.round((total_frag_size / total_repeat), 2)
        frag_size = np.round((total_frag_size / unique_repeat), 2)
        ##########################
        # smis_lst_dct = {}
        # for i in range(len(smis_and_losses)):
            # temp77_smi = smis_and_losses[i][0]
            # temp77_loss = smis_and_losses[i][1] 
            # # just_smis.append(temp77_smi)
            # # just_test_losses.append(temp77_loss)
            # smis_lst_dct[temp77_smi] = temp77_loss
        ##########################
        # frag_size = frag_size_dict[sub_smi]
        # frag_size = int(frag_size)
        ##########################
        if unique_repeat < 2:
            continue
        ##########################
        x_lst.append(frag_size) 
        y_lst.append(unique_repeat)
        z_lst.append(total_point)
        sub_smis.append(sub_smi)
        # w_lst.append(total_repeat)
        subs2smis_dct[sub_smi] = smis_and_losses
        # cids.append(cid)
        # names.append(name)
    ##########################
    xyz["x"] = x_lst
    xyz["y"] = y_lst
    xyz["z"] = z_lst
    xyz["sub_smis"] = sub_smis
    # xyz["w"] = w_lst
    xyz["smis"] = subs2smis_dct
    # xyz["cid"] = cids
    # xyz["name"] = names
    ##########################
    return xyz, frag_size_dict

######################################################################################## 

def plot_contour(all_subtrees, repeat_dict, args):   
    ####################################################
    xyz, _ = set_xyz(all_subtrees, repeat_dict)
    ##########################
    x = xyz["x"]   # fragment sizes
    y = xyz["y"]   # unique_repeat
    z = xyz["z"]   # total_point
    sub_smis = xyz["sub_smis"]
    # tr = xyz["w"]   # total_repeat
    smis = xyz["smis"]   # parent smiles and test loss dict
    ur = y
    ##########################
    if args.repeat_type == "ur":
        y_label = "\nloge(Unique Repeat Count)\n"
    elif args.repeat_type == "tr":
        y_label = "\nloge(Total Repeat Count)\n"
    ##########################
    x = np.asarray(x)
    x = np.log(x).tolist()
    ##########################
    y = np.asarray(y)
    y = np.log(y).tolist()
    ##########################
    z = np.asarray(z)
    z = z / np.max(z)
    z = z.tolist()
    # cids = xyz["cid"]
    # names = xyz["name"]
    ##########################
    thr2 = args.task_avg_loss / np.max(z)
    ####################################################
    z, x, y, sub_smis, ur = zip(*reversed(sorted(zip(z, x, y, sub_smis, ur))))
    #################################################### 
    # if args.xt == 0:
        # if args.yt == 0:
            # args.xt = 1
            # args.yt = 1
    ####################################################
    x_ticks = np.arange(0, (max(x) + args.xt), args.xt, dtype=int).tolist()
    # x_ticks.append(max(x))
    # x_ticks.append(50)
    # x_ticks = sorted(x_ticks)
    ##########################
    cb_tick_lst = np.arange(0, max(z), args.cbr).tolist()
    cb_tick_lst.append(max(z))
    ##########################
    y_ticks = np.arange(0, (max(y) + args.yt), args.yt, dtype=int).tolist()
    # y_ticks.append(max(y))
    ####################################################
    sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="white")
    x_good, y_good = [], []
    # x_bad, y_bad = [], [] 
    z_good = {}
    cids = []
    save_loc = (f"{args.save_dir}/{args.data_name}")
    print("\n")
    ########################################################################################################
    ########################################################################################################
    cnt = 0
    mols = []
    mcidz = []
    print(f">> {args.data_name.upper()} fragments are being visualized..\n")
    #########################
    for i in range(len(z)):
        zi = np.round(z[i], 4)
        cnt += 1
        ####################################################
        # if args.task == "reg":
            # if zi > (thr2 / 2):   # mean of the task losses
                # continue
        #########################
        # elif args.task == "clf":
            # if int(zi) != 0:
                # continue
        ####################################################
        if cnt <= args.thr:
            ##########################
            x_good.append(x[i])
            y_good.append(y[i])
            sub_smi = sub_smis[i]
            z_good[sub_smi] = i
            rank = z[i]
            ##########################
            c = pcp.get_compounds(sub_smis[i], "smiles")[0]
            cid_str = (f"CID {c.cid}")
            cid_str2 = (f"{(i + 1)} - {cid_str}")
            cid_str_annot = (f"{cid_str} ({zi:.4f})")
            cids.append([cid_str_annot, x[i], y[i]])
            ##########################
            m = Chem.MolFromSmiles(sub_smi)
            mols.append(m)
            mcidz.append(cid_str2)
            ##########################
            parents = pd.DataFrame(list(smis[sub_smi].keys()))
            # parents = pd.DataFrame.from_dict(smis[sub_smi])
            parents.columns = ["smiles"]
            parents.to_csv(f"{save_loc}/csvs/{(i + 1)} - CID {c.cid}.csv")
            ##########################
            fig = Draw.MolToMPL(m, size=(350, 350))
            title = (f"{c.iupac_name}\n{cid_str}\n{sub_smi}\n\nUR = {ur[i]}    FS = {x[i]:.2f}     TP = {zi:.4f}\nTask = {args.data_name.upper()}     Rank = {(i + 1)}")
            fig.suptitle(title, fontsize=35, x=1.25, y=0.8)
            fig.set_size_inches(5.5, 5.5)
            ##########################
            if not os.path.exists(save_loc):
                os.mkdir(save_loc)
            ##########################
            # save_name = (f"{save_loc}/images/{args.repeat_type.upper()} {(i + 1)} - {cid_str}.png")
            save_name = (f"{save_loc}/images/{(i + 1)} - {cid_str}.png")
            plt.savefig(fname=save_name, bbox_inches="tight", dpi=100)
            plt.close(fig)
            print(f"{cnt} -> {cid_str_annot}")
            ####################################################
        else:
            # x_bad.append(x[i])
            # y_bad.append(y[i])
            pass
        ####################################################
    print("\n")
    ########################################################################################################
    ########################################################################################################
    fig_grid = Draw.MolsToGridImage(mols, 
                                    legends=mcidz, 
                                    molsPerRow=4,
                                    subImgSize=(400, 400))
    save_name = (f"{save_loc}/images/{args.data_name.upper()} All Fragments.png")
    fig_grid.save(save_name, format="PNG")
    ########################################################################################################
    sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")
    fig, ax = plt.subplots()
    plt.tricontourf(x, y, z, args.contour_level_fill, cmap="CMRmap_r", zorder=1)   # "RdBu_r", "Spectral_r"
    plt.colorbar(ticks=cb_tick_lst, label="\nImportance Metric\n= (avg frag size * avg frag test loss) / max frag size\n")
    # scatter_bad = ax.scatter(x_bad, y_bad, marker="o", c="white", zorder=2)
    contours = plt.tricontour(x, y, z, args.contour_level_line, linewidths=3, colors="black", zorder=3)
    # ax.axvline(50, ymax=max(y), linewidth=3.5, c="black", zorder=4)
    ax.clabel(contours, inline=True, fontsize=20, zorder=5)
    scatter_good = ax.scatter(x_good, y_good, marker="X", s=300, c="white", edgecolors="black", zorder=6)
    # ax.scatter(x_good, y_good, marker="o", s=1500, linewidth=5, facecolors="none", edgecolors="black", zorder=7)
    ####################################################
    # for i in range(len(cids)):
        # ax.annotate(cids[i][0], ((cids[i][1] + 0.15), (cids[i][2] + 4)), fontsize=35, zorder=7)
    # ax.legend(
              # # *scatter_good.legend_elements(),
              # ("X", "o"),
              # # ("Goods", "Bads"),
              # loc="upper right",
              # fontsize=50,
              # )
    # # ax.add_artist(legend)
    ####################################################
    plt.title(f"\nFragment Importance Contour Plot\n({args.data_name.upper()} Task - {len(z)} fragments)\n\n", fontsize=70)
    plt.ylabel(y_label)   # , fontsize = ?
    plt.xlabel("\nloge(Fragment Size)\n")   # , fontsize = ?
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    # plt.ylim(0, int(np.floor(max(y))))
    # plt.xlim(0, int(np.floor(max(y))))
    ####################################################
    # save_name = (f"{save_loc}/images/{args.repeat_type.upper()} importance map.png")
    save_name = (f"{save_loc}/images/{args.data_name.upper()} Importance Map.png")
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






























































