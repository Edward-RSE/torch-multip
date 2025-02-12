#!/bin/bash

# I am requesting here an extra CPU per task because I have set it such that
# each dataloader, when using CUDA, has one additional worker to do the
# data loading.

#SBATCH --partition=swarm_a100
#SBATCH --gpus=2
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --threads-per-core=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --mem=32G
#SBATCH --time=00:10:00

export PYTHONUNBUFFERED=1

RESULTS_DIR=$(pwd)

function start_nvidia_smi() {
    nvidia-smi --list-gpus
    nvidia-smi --query --loop=$1 --xml-format --filename=$RESULTS_DIR/$SLURM_JOB_NAME.$SLURM_JOB_ID.nvidia-smi.xml &
    NVIDIA_SMI_PID=$!
}

function stop_nvidia_smi() {
    kill -SIGTERM $NVIDIA_SMI_PID
}

function process_nvidia_smi() {
    sed -i '/^[[:space:]]*$/d' $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
    sed -i '1!{/^<?xml version="1.0" ?>/d;}' $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
    sed -i '1!{/^<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">/d;}' $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
    sed -i '/<?xml version="1.0" ?>/a <nvidia_smi>' $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
    sed -i -e '$a</nvidia_smi>' $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
    xsltproc --output $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.csv nvidia-smi.xml2csv.xslt $RESULTS_DIR/"$SLURM_JOB_NAME"."$SLURM_JOB_ID".nvidia-smi.xml
}

source $HOME/.pyenv/versions/ctdcomm-3.12.8/bin/activate

start_nvidia_smi 2

# Using num-workers=1 so each torchrun process only spawns a single process.
torchrun --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_NTASKS_PER_NODE main.py \
    --multinode \
    --use-cuda \
    --num-epochs 10

stop_nvidia_smi
process_nvidia_smi
