#!/bin/bash
#SBATCH -J TransFusion
#SBATCH --time=03:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

curl -d "Started TransFusion on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig

DATASET_NO=$SLURM_ARRAY_TASK_ID

module purge
pip freeze --user | xargs pip uninstall -y

module load PyTorch/1.12.1-foss-2021b-CUDA-11.5.2 
pip install pandas scikit-learn tqdm scipy einops tensorboard numba pyprojroot mgzip

python train.py $DATASET_NO

curl -d "Finished TransFusion on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
