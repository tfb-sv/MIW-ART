#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J esol
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=49:59:59
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/esol.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "esol" --is_cv "besty" --task "reg" --proj_name "best" 