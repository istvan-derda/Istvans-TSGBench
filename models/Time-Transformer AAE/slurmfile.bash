#!/bin/bash
#SBATCH -J Time-Transformer-AAE
#SBATCH --time=03:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --mem=4G
#SBATCH -o jobfiles/log/%x.out-%j

DATASET_NO=$SLURM_ARRAY_TASK_ID

curl -d "Started Time-Transformer AAE on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig

source env/sc_uni_leipzig/init.bash
python run_time_transformer.py --dataset_no $DATASET_NO

curl -d "Finished Time-Transformer AAE on Dataset D$DATASET_NO" ntfy.sh/istvanshpcunileipzig
