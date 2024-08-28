#!/bin/bash
#SBATCH -J sst-1000self
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00

module add cuda/10.2
module add anaconda
source activate
conda activate myenv

CUDA_VISIBLE_DEVICES=0,1 python CL-self.py
