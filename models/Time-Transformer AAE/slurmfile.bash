#!/bin/bash
#SBATCH -J Time-Transformer-AAE
#SBATCH --time=03:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err


DATASET_NO=$SLURM_ARRAY_TASK_ID

curl -d "Started Time-Transformer AAE on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig

source env/sc_uni_leipzig/setup_env
python run_time_transformer.py --dataset_no $DATASET_NO

curl -d "Finished Time-Transformer AAE on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig
