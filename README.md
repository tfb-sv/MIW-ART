# Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)

Here are some example commands below to run the related scripts. These can be used in Anaconda prompt.

## Firstly, indicate the environment and the project folder directory:

> activate yourEnv

> cd C:\yourFolders\MIW-ART

## For training:

> python train_tree.py --data_name "freesolv"

## For testing the model, obtaining the Newick strings or visualizing the tree structures:

You need to run firstly "training" OR you can use an available model checkpoint which should be in "results/training_results" folder.

> python eval_tree.py --mode "test" --data_names "freesolv"

> python eval_tree.py --mode "newick" --data_names "freesolv"

> python eval_tree.py --mode "visualize" --data_names "freesolv"

## For inspecting the molecular fragments:

You need to run firstly "training" OR you can use an available model checkpoint which should be in "results/training_results" folder.

You need to run secondly "Newicking" OR you can use an available Newick file which should be in "results/evaluation_results" folder.

> python inspect_subtrees.py --data_names "freesolv"

## All dataset names for the commands:

"bace_clf", "bbbp", "clintox", "tox21", "lipo", "esol", "freesolv", "bace_reg", "sider"

## Other Informations

There are lots of arguments in the related scripts but does not need to any change to run the scripts.
