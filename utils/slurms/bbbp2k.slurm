#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J bbbp2k
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=49:59:59
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/bbbp2k.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "bbbp2k" --is_cv "besty" --proj_name "B1"