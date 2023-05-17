########################################################################################
########################################################################################

from ete3 import Tree
import re
import string

########################################################################################
########################################################################################

def tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    smi_new = "".join(tokens)
    if smi != smi_new:
        print(f"\n{smi}\n{smi_new}")
    assert smi == smi_new
    return tokens

########################################################################################

def decodeTokens(enc_smis, decoder):
    ####################################################
    enc_smis2 = enc_smis.split("x")
    enc_smis2.reverse()
    decoded_lst = [decoder[int(e)] for e in enc_smis2]
    final_smi = "".join(decoded_lst)
    ##########################
    return final_smi, decoded_lst
    ####################################################

########################################################################################

def recoverTree(main_newick, decoder, orj_smi):
    ####################################################
    is_okay = True
    ##########################
    t = Tree(main_newick, format=8)
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    ####################################################
    for i in reversed(range(int(max_len))):
        ####################################################
        for node in t.traverse():
            tmp_dist = t.get_distance(root, node)
            ##########################
            if tmp_dist == i:
                ##########################
                if node.children == []:
                    continue
                ##########################
                child_right = node.children[0]
                child_left = node.children[1]
                ##########################
                if child_right.name == "-":
                    child_right.name = ""
                else:
                    child_right.name = "x" + child_right.name
                ##########################
                if child_left.name == "-":
                    child_left.name = ""
                else:
                    child_left.name = child_left.name + "x"
                ##########################
                ##########################
                node.name = child_left.name + node.name + child_right.name
                node.name = node.name.replace("xx", "x")
                ##########################
                child_right.name = "-"
                child_left.name = "-"
                ##########################
        ####################################################
    if root.name.startswith("x"):
        root.name = root.name[1:]
    if root.name.endswith("x"):
        root.name = root.name[:-1]
    ####################################################
    final_smi, decoded_lst = decodeTokens(root.name, decoder)
    ##########################
    if final_smi != orj_smi:    
        is_okay = False
        # print("\n", final_smi, "\n", orj_smi, "\n")
    ##########################
    return decoded_lst, is_okay
    ####################################################

########################################################################################

def decodePoints(enc_smis, decoder):
    ####################################################
    enc_smis2 = enc_smis.split("x")
    enc_smis2.reverse()
    ##########################
    return enc_smis2
    ####################################################

########################################################################################

def recoverPoints(main_newick, decoder, orj_smi):
    ####################################################
    t = Tree(main_newick, format=8)
    root = t.get_tree_root()
    _, max_len = root.get_farthest_leaf()
    ####################################################
    for i in reversed(range(int(max_len))):
        ####################################################
        for node in t.traverse():
            tmp_dist = t.get_distance(root, node)
            ##########################
            if tmp_dist == i:
                ##########################
                tmp_prnt_point = (max_len + 1) - tmp_dist
                tmp_chld_point = tmp_prnt_point - 1
                ##########################
                if node.children == []:
                    continue
                ##########################
                child_right = node.children[0]
                child_left = node.children[1]
                ##########################
                if child_right.name == "-":
                    child_right.name = ""
                else:
                    child_right.name = "x" + str(tmp_chld_point)
                ##########################
                if child_left.name == "-":
                    child_left.name = ""
                else:
                    child_left.name = str(tmp_chld_point) + "x"
                ##########################
                ##########################
                node.name = child_left.name + str(tmp_prnt_point) + child_right.name
                node.name = node.name.replace("xx", "x")
                ##########################
                child_right.name = "-"
                child_left.name = "-"
                ##########################
                # t_ascii = t.get_ascii(format=8)
                # print(t_ascii, "\n")
        ####################################################
    if root.name.startswith("x"):
        root.name = root.name[1:]
    if root.name.endswith("x"):
        root.name = root.name[:-1]
    ####################################################
    point_lst = decodePoints(root.name, decoder)
    ##########################
    return point_lst
    ####################################################

########################################################################################

def deal_with_atoms(smi):
    new_smi = ""
    for i in range(len(smi)):
        cha = smi[i]
        if cha not in string.ascii_letters:
            continue
        new_smi += cha
    return new_smi

########################################################################################
########################################################################################
































































