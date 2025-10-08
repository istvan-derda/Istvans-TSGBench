#!/bin/bash
#SBATCH -J TSGBench_Evaluation
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

methods=("JustCopy" "TimeGAN" "TTS-GAN" "Time-Transformer" "TransFusion" "TimeVQVAE")

method=${methods[$SLURM_ARRAY_TASK_ID]}

echo "running slurm script to evaluate $method"

eval "$(conda shell.bash hook)"
conda activate tsgbench

python evaluate.py $method
