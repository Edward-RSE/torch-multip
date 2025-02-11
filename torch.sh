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

python main.py \
    distributed \
    --dist-backend nccl \
    --use-cuda \
    --world-size 2 \
    --num-epochs 10

stop_nvidia_smi
process_nvidia_smi
