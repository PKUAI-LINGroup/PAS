#!/bin/bash
#SBATCH -J 1000pre+fine+self
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00

module add cuda/10.2
module add anaconda
source activate
conda activate py3.7

CUDA_VISIBLE_DEVICES=0,1 python CL-pre+fine+self.py
