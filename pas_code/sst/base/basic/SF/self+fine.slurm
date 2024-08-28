#!/bin/bash

#SBATCH -J sst-self+fine
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00

module add cuda/10.2
module add anaconda
source activate
conda activate py3.7

CUDA_VISIBLE_DEVICES=0 python self+fine.py
