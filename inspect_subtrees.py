# with open("../data/datasets/id2prot.json", "r") as f:
#     id2prot = json.load(f)

from ete3 import Tree

####################################################

# GE-BDAC ?????

##########################

# nwck = "(C,(D,(-,G)E)B)A;"
# t = Tree(nwck, format=8)
# t_ascii = t.get_ascii(show_internal=True)
# print(t_ascii)

####################################################

nodes = [node for node in t.traverse()]
subtrees = []

for i in range(len(nodes)):
    node = nodes[i]
    if i == 0:
        continue
    if node.is_leaf():
        continue
    subtree_newick = node.write(format=8)
    subtrees.append(subtree_newick)
    # subtree_ascii = node.get_ascii(show_internal=True)
    # print(subtree_ascii)