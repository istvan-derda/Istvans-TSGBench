#!/bin/bash
#SBATCH -J Time-Transformer-AAE
#SBATCH --time=03:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

log_stdout_stderr() {
    echo "$@" | tee >(cat >&2)
}

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] [NOTICE] $*"
    local border
    border=$(printf '%*s' "${COLUMNS:-80}" '' | tr ' ' '#')

    log_stdout_stderr "$border"
    log_stdout_stderr "$msg"
    log_stdout_stderr "$border"
}

DATASET_NO=$SLURM_ARRAY_TASK_ID

curl -d "Started Time-Transformer AAE job on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig
log "Started Time-Transformer AAE job on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)"

log "Resetting environment"
module purge

log "Setting up environment"
module load TensorFlow
VENV_DIR=/tmp/slurm_job_${SLURM_JOB_ID}_venv
python -m venv --system-site-packages $VENV_DIR
source $VENV_DIR/bin/activate
pip install numpy keras==2.11.0 scikit-learn==1.0.2 scipy==1.7.3 mgzip==0.2.1

log "Starting Training Script"
python run_time_transformer.py --dataset_no $DATASET_NO
log "Training Script Terminated"

deactivate
module purge

curl -d "Finished Time-Transformer AAE on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig
log "Finished Time-Transformer AAE on Dataset D$DATASET_NO"
