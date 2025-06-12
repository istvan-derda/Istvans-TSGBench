#!/bin/bash
#SBATCH -J TimeGAN
#SBATCH --time=1-00:00:00
#SBATCH --array=2-7
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --mem=4G
#SBATCH -o jobfiles/log/%x.out-%j
#SBATCH -e jobfiles/log/%x.err-%j

DATASET_NO=$SLURM_ARRAY_TASK_ID

curl \
	-H "Email: wi34sasa@studserv.uni-leipzig.de" \
	-d "Started TimeGAN on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig

source env/sc_uni_leipzig/init.bash
python run_time_gan.py --dataset_no $DATASET_NO

curl \
	-H "Email: wi34sasa@studserv.uni-leipzig.de" \
	-d "Finished TimeGAN on Dataset D$DATASET_NO (Job ID: $SLURM_JOB_ID)" ntfy.sh/istvanshpcunileipzig
