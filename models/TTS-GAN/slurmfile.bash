#!/bin/bash
#SBATCH -J TTS-GAN
#SBATCH --time=2-00:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

log_stdout_stderr() {
    echo "$@" | tee >(cat >&2)
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

curl -d "Started TTS_GAN job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
log_progress "Started TTS_GAN job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)"

log_progress "Setting up environment"
module purge
module load CUDA/12.6.0
module load Python/3.10.8-GCCcore-12.2.0
pip install -r requirements.txt


log_progress "Starting Training Script"
python run_tts_gan.py --dataset_no $DATASET_NO --exp_name D$DATASET_NO
log_progress "Training Script Terminated"

curl -d "Finished TTS_GAN job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
log_progress "Finished TTS_GAN job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)"
