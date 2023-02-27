#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J hiv
#SBATCH -n 1
#SBATCH -c 20
#SBATCH -p akya-cuda
#SBATCH --time=359:59:59
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/hiv.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "hiv" --is_cv "feasible"