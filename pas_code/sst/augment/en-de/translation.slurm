#!/bin/bash

#SBATCH -J translation
#SBATCH --cpus-per-task=8
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -t 12:00:00

module add cuda/10.2
module add anaconda
source activate
conda activate py3.7

bash preprocess.sh
bash generate.sh
grep ^H /mnt/lustrefs/home/wangyh/sst/augment/en-de/result.txt | sort -n -k 2 -t '-' | cut -f 3 > sstback.de