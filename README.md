[![License: MIT][mit-shield]](./LICENSE)

# Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)
This repository contains the implementation of the MIW-ART paper.

## Overview
- Identify and interpret the significant drug molecule fragments for the corresponding molecular property tasks (e.g., physiology, biophysics, and physical chemistry), a sequence of processes is proposed named **Molecular Interpretation Workflow through Attentive Recursive Tree Model (MIW-ART)** by utilizing the [Attention Recursive Tree (AR-Tree)](https://github.com/shijx12/AR-Tree) model.

## Dataset
- **MoleculeNet**: A benchmark dataset collection for various molecular property prediction and classification tasks.

## Setup
Install required libraries:

> cd <your_directory>\MIW-ART\code

> pip install -r requirements.txt

## Usage

### Training:
To train the model from scratch.

> python train_tree.py --data_name "freesolv"

### Evaluation:
To evaluate the model checkpoints, obtain the Newick strings, or visualize the tree structures.

Firstly, you need to run the `train_tree.py` script **OR** you can use an available model checkpoint, which is in the `results/training_results` folder.

> python eval_tree.py --mode "test" --data_names "freesolv"

> python eval_tree.py --mode "newick" --data_names "freesolv"

> python eval_tree.py --mode "visualize" --data_names "freesolv"

### Inspection:
To inspect the molecular fragments.

Firstly, you need to run the `newick` mode of the `eval_tree.py` script **OR** you can use an available Newick file, which is in the `results/evaluation_results` folder.

> python inspect_subtrees.py --data_names "freesolv"

### All dataset names for the commands:
`bace_clf`, `bbbp`, `clintox`, `tox21`, `lipo`, `esol`, `freesolv`, `bace_reg`, `sider`

## Additional Information
- The complete contents (around 1.5 GB) of the `results` folder can be accessed through [this link](https://drive.google.com/drive/folders/1A1q138vF3G-SG-aRxiA8LbI04OZk6H7w?usp=sharing), simply replacing the `results` folder is enough.
- There are lots of arguments in the related scripts, but no changes are needed to run them.
- The folders in the `results/training_results`, the `results/evaluation_results`, and the `results/inspection_results` are overwritten with each related run, so be careful about this. Also, if the folders within these three results directories do not exist, they will be automatically created.
- Note that the order of fragments in the `results/inspection_results` folder may vary slightly, even when using the same model checkpoints as provided by us.
- Retrieving all the results in the `results` folder from scratch, specifically the **training** process of all the tasks, can take **days**. Please keep this in mind.

## License
© 2025 [Nural Ozel](https://github.com/tfb-sv).

- This work is licensed under the [MIT License](./LICENSE).
- The **data** associated with this work is licensed under a [CC0 1.0 Universal](./data/LICENSE).
- The **paper** associated with this work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)][cc-by].

**MIW-ART** uses the following open-source work:
- [AR-Tree](https://github.com/shijx12/AR-Tree) by [shijx12](https://github.com/shijx12)


[cc-by]: https://creativecommons.org/licenses/by/4.0/
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
