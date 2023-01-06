########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################

from ete3 import Tree

########################################################################################

nwck = "((F,G)B,(D,(-,H)E)C)A;"   # "ABC;"   # 
# olması gereken = "HECDAGBF"
t = Tree(nwck, format=8)
t_ascii = t.get_ascii(show_internal=True)
# print(t_ascii)
# nodes = [node for node in t.traverse()]

########################################################################################

# root = t.get_tree_root()
# print(t_ascii, "\n")
# for node in t.traverse():
#     if node == root:
#         continue
#     parent = node.up
#     print(parent.name, parent.children[0].name, parent.children[1].name, ">>", node.name)

########################################################################################

root = t.get_tree_root()
_, max_len = root.get_farthest_leaf()
print(t_ascii)
while max_len > 0.0:
    for node in t.traverse():
        if not node.is_leaf():   # leaf değilse
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

########################################################################################























