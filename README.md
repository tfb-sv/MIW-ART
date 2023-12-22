# Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)

Here are some example commands below to run the related scripts. These can be used in Anaconda prompt. Also, various related files with environmental setup can be found in the "settings" folder.

## Firstly, indicate the environment and the project directory:

The following two commands are for example purposes, you should modify these commands according to your needs.

> conda activate <yourEnvironment>

> cd C:\<yourPreDirectory>\MIW-ART\code

## For training:

> python train_tree.py --data_name "freesolv"

## For testing the model checkpoints, obtaining the Newick strings, or visualizing the tree structures:

Firstly, you need to run the "train_tree.py" script OR you can use an available model checkpoint, which should be in "output/training_results" folder.

> python eval_tree.py --mode "test" --data_names "freesolv"

> python eval_tree.py --mode "newick" --data_names "freesolv"

> python eval_tree.py --mode "visualize" --data_names "freesolv"

## For inspecting the molecular fragments:

Firstly, you need to run the "newick" mode of the "eval_tree.py" script OR you can use an available Newick file, which should be in "output/evaluation_results" folder.

> python inspect_subtrees.py --data_names "freesolv"

## All dataset names for the commands:

"bace_clf", "bbbp", "clintox", "tox21", "lipo", "esol", "freesolv", "bace_reg", "sider"

## Other Informations

The complete contents (around 1.5 GB) of the "output" folder can be accessed through the link below. Simply replacing the "output" folder is enough.

https://drive.google.com/drive/folders/1A1q138vF3G-SG-aRxiA8LbI04OZk6H7w?usp=sharing

There are lots of arguments in the related scripts, but no changes are needed to run them.

The folders in the "output/training_results", the "output/evaluation_results", and the "output/inspection_results" are overwritten with each related run, so be careful about this. Also, if the folders within these three output directories do not exist, they will be automatically created.

Note that the order of fragments in the "output/inspection_results" folder may vary slightly, even when using the same model checkpoints as provided by us.

Retrieving all the outputs in the "results" folder from scratch, specifically the "training" process of all the tasks, can take days. Please keep this in mind.

The publication related to this work will be made available here upon its release.
