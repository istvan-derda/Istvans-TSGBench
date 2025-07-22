#!/bin/bash
#SBATCH -J TransFusion
#SBATCH --time=03:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

log_stdout_stderr() {
    echo "$@" | tee /dev/stderr
}

log_progress() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [NOTICE] $*"
    local border
    border=$(printf '%*s' "${COLUMNS:-80}" '' | tr ' ' '#')

    log_stdout_stderr "$border"
    log_stdout_stderr "$msg"
    log_stdout_stderr "$border"
}

DATASET_NO=$SLURM_ARRAY_TASK_ID

curl -d "Started TransFusion job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
log_progress "Started TransFusion job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)"

log_progress "Resetting environment"
module purge
pip freeze --user | xargs pip uninstall -y

log_progress "Setting up environment"
module load PyTorch/1.12.1-foss-2021b-CUDA-11.5.2 
pip install pandas scikit-learn tqdm scipy einops tensorboard numba pyprojroot mgzip

log_progress "Starting Training Script"
python train.py $DATASET_NO
log_progress "Training Script Terminated"

curl -d "Finished TransFusion job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
log_progress "Finished TransFusion job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" | tee /dev/stderr

