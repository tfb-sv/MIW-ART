#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J bace
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=359:59:59
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/bace.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "bace" --is_cv "besty" --proj_name "ideal"