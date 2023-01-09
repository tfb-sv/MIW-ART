#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J hiv
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/hiv.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "hiv" --is_cv "feasible"