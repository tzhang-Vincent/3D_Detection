#!/bin/bash
#SBATCH --job-name=hostname
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --output=%j.out
#SBATCH --error=&j.err
module load anaconda/3
module load cuda/8.0
module load cudnn/6.0
source activate py36
python3 train/train.py --gpu 0 --model frustum_pointnets_v2 --log_dir train/log_v2 --num_point 1024 --max_epoch 201 --batch_size 24 --decay_step 800000 --decay_rate 0.5