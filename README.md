# Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)

Here are some example commands below to run the related scripts. These can be used in the Anaconda prompt. Also, various files related to environmental setup can be found in the `settings` folder.

## Firstly, indicate the environment and the project directory:

The following two commands are for example purposes, you should modify these commands according to your needs.

> conda activate <env_name>

> cd <project_root>\MIW-ART\code

## For training:

> python train_tree.py --data_name "freesolv"

## For testing the model checkpoints, obtaining the Newick strings, or visualizing the tree structures:

Firstly, you need to run the `train_tree.py` script **OR** you can use an available model checkpoint, which is in the `results/training_results` folder.

> python eval_tree.py --mode "test" --data_names "freesolv"

> python eval_tree.py --mode "newick" --data_names "freesolv"

> python eval_tree.py --mode "visualize" --data_names "freesolv"

## For inspecting the molecular fragments:

Firstly, you need to run the `newick` mode of the `eval_tree.py` script **OR** you can use an available Newick file, which is in the `results/evaluation_results` folder.

> python inspect_subtrees.py --data_names "freesolv"

## All dataset names for the commands:

`bace_clf`, `bbbp`, `clintox`, `tox21`, `lipo`, `esol`, `freesolv`, `bace_reg`, `sider`

## Other Informations

- The complete contents (around 1.5 GB) of the `results` folder can be accessed through [this link](https://drive.google.com/drive/folders/1A1q138vF3G-SG-aRxiA8LbI04OZk6H7w?usp=sharing), simply replacing the `results` folder is enough.

- There are lots of arguments in the related scripts, but no changes are needed to run them.

- The folders in the `results/training_results`, the `results/evaluation_results`, and the `results/inspection_results` are overwritten with each related run, so be careful about this. Also, if the folders within these three results directories do not exist, they will be automatically created.

- Note that the order of fragments in the `results/inspection_results` folder may vary slightly, even when using the same model checkpoints as provided by us.

- Retrieving all the results in the `results` folder from scratch, specifically the **training** process of all the tasks, can take **days**. Please keep this in mind.

- The publication related to this work will be made available here upon its release.

## License

- The **code** in this repository is licensed under the [MIT License](./LICENSE).
- The **data** under `data/` is licensed under the [CC0 1.0 Universal](./data/LICENSE).
- The **preprint** associated with this project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
