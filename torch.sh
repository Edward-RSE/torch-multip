#!/bin/bash

#SBATCH --partition=swarm_l4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00

source $HOME/.pyenv/versions/ctdcomm-3.12.8/bin/activate

python main.py \
    distributed \
    --dist-backend nccl \
    --use-cuda \
    --world-size 1 \
    --num-epochs 10
