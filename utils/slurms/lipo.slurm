#!/bin/bash
#SBATCH -A tbag80
#SBATCH -J lipo
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p akya-cuda
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/lipo.log

~/miniconda3/envs/AR-TreeDTA/bin/python -u train_tree.py --data_name "lipo" --is_cv "besty" --proj_name "B4"