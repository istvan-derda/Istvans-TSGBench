#!/bin/bash
#SBATCH -J Time-Transformer-AAE
#SBATCH --time=03:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100
#SBATCH --mem=2G
#SBATCH -o $HOME/time-transformer.out-%a-%A
#SBATCH -e $HOME/time-transformer.err-%a-%A

DATASET_NO=$(( $SLURM_ARRAY_TASK_ID + 1 ))

python run_time_transformer.py --dataset_no $DATASET_NO