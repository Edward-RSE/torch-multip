#!/bin/bash

# There is 1 worker per dataloader per GPU process, I think. So we should
# request more CPUs than tasks.

#SBATCH --partition=swarm_l4
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem=32G
#SBATCH --time=00:10:00

RESULTS_DIR=$(pwd)

function start_nvidia_smi() {
    nvidia-smi --list-gpus
    nvidia-smi --query --loop=$1 --xml-format --filename=$RESULTS_DIR/$SLURM_JOB_NAME.$SLURM_JOB_ID.nvidia-smi.xml &
    NVIDIA_SMI_PID=$!
}

function stop_nvidia_smi() {
    kill -SIGTERM $NVIDIA_SMI_PID
}

source $HOME/.pyenv/versions/ctdcomm-3.12.8/bin/activate

start_nvidia_smi 2

python main.py \
    distributed \
    --dist-backend nccl \
    --use-cuda \
    --world-size 2 \
    --num-epochs 10

stop_nvidia_smi
