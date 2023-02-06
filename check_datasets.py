########################################################################################
########################################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

########################################################################################
########################################################################################

from chembench import load_data, dataset, get_cluster_induces

########################################################################################
########################################################################################

def count_dfs(mine_df, other_df):
    cnt_pos, cnt_neg = 0, 0
    for smi in other_df["smiles"]:
        if smi in mine_df["smiles"]:
            cnt_pos += 1
        else:
            cnt_neg += 1
    print("\n", cnt_pos, cnt_neg)
    return cnt_pos, cnt_neg

########################################################################################
########################################################################################
########################################################################################
# Usage-1: Load the Dataset and MoleculeNet's Split Induces

df, induces = load_data("BBBP")
# get the 3 times random split induces
train_idx, valid_idx, test_idx = induces[0]
train_idx, valid_idx, test_idx = induces[1]
train_idx, valid_idx, test_idx = induces[2]

########################################################################################
########################################################################################
# Usage-2: Load Dataset As Data Object

data = dataset.load_ESOL()
data.x
data.y
data.description

##########################

dataset.load_BBBP()   # scaffold-3, roc
dataset.load_BACE()   # scaffold-3, roc
dataset.load_HIV()   # scaffold-3, roc
dataset.load_Tox21()   # random-3, roc
dataset.load_ClinTox()   # random-3, roc

########################################################################################
########################################################################################
# Usage-3: Load Cluster Splits

induces1 = get_cluster_induces("BBBP", induces="random_5fcv_5rpts")   # "random_5fcv_5rpts", "scaffold_5fcv_1rpts"
induces2 = get_cluster_induces("BBBP", induces="scaffold_5fcv_1rpts")
print(len(induces1))
print(len(induces2))

########################################################################################
########################################################################################
########################################################################################

import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_tox21(split='random', featurizer = 'Raw')
train, valid, test = datasets
train.x #smiles list of training set
train.y #targets

########################################################################################

# https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html

.load_tox21()
.load_hiv()
.load_clintox()
.load_bbbp()
.load_bace_classification()

########################################################################################
########################################################################################
########################################################################################
# sözde moleculenet'i yazan kişinin çözümü

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem as dc
import pickle
import tempfile
from deepchem.molnet.run_benchmark_models import benchmark_classification
from deepchem.molnet.run_benchmark import load_dataset
from deepchem.molnet.check_availability import CheckFeaturizer

# Example
dataset = 'tox21'
split = 'random'
model = 'tf'

pair = (dataset, model)
if pair in CheckFeaturizer:
  featurizer = CheckFeaturizer[pair][0]
  n_features = CheckFeaturizer[pair][1]

# Load all data
tasks, all_dataset, transformers = load_dataset(dataset, featurizer, split='index')
all_dataset = dc.data.DiskDataset.merge(all_dataset)

# Read index of split
seed = 123
with open(dataset + split + str(seed) + '.pkl', 'r') as f:
  inds = pickle.load(f)
train = all_dataset.select(inds[0], tempfile.mkdtemp()) # Rebuild train set
valid = all_dataset.select(inds[1], tempfile.mkdtemp()) # Rebuild valid set
test = all_dataset.select(inds[2], tempfile.mkdtemp()) # Rebuild test set

# Benchmark run
metric = [dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode='classification')]
train_score, valid_score, test_score = benchmark_classification(
    train, valid, test, tasks, transformers, n_features, metric,
    model, test=True, seed=seed)

print(train_score)
print(valid_score)
print(test_score)    

########################################################################################
########################################################################################
########################################################################################












