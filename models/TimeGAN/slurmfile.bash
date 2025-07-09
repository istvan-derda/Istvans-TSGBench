#!/bin/bash
#SBATCH -J TimeGAN
#SBATCH --time=2-00:00:00
#SBATCH --array=3,4,6,7
#SBATCH --partition=clara
#SBATCH --mem=16G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

DATASET_NO=$SLURM_ARRAY_TASK_ID

curl \
	-d "Started TimeGAN on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig

eval "$(conda shell.bash hook)"
conda activate time-gan
python run_time_gan.py --dataset_no $DATASET_NO

curl \
	-d "Finished TimeGAN on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
