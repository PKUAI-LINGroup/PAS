#!/bin/bash
#SBATCH -J elec-pre+self
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00

module add cuda/10.2
module add anaconda
source activate
conda activate py3.7

CUDA_VISIBLE_DEVICES=0,1 python CL-pre+self.py
