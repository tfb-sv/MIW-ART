# Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)

Here are some example commands below to run the related scripts. These can be used in Anaconda prompt. Also, the necessary libraries can be found in "environment.yml" file.

## Firstly, indicate the environment and the project directory:

> conda activate yourEnv

> cd C:\yourDirectory\MIW-ART

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

The complete contents of the "results" folder can be accessed through this link: ""

There are lots of arguments in the related scripts, but no changes are needed to run them.

The folders in "results/training_results", "results/evaluation_results", and "results/inspection_results" are overwritten with each related run, so be careful about this. Also, if the folders within these three result directories do not exist, they will be automatically created.

The publication related to this work will be made available here upon its release.

