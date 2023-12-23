set -ex

# The entire process below can take days, please keep this in mind.

for dataset in bace_clf bbbp clintox tox21 lipo esol freesolv bace_reg sider
do
    python -u train_tree.py "$@" --data_name "$dataset"
    python -u eval_tree.py "$@" --mode "newick" --data_name "$dataset"
    python -u inspect_subtrees.py "$@" --data_name "$dataset"
done
