#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J bbbp8k
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=359:59:59
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/bbbp8k.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "bbbp8k" --is_cv "feasible" --y_label "affinity_score"