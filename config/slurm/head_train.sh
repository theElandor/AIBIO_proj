#!/bin/sh
#SBATCH -e /homes/mlugli/output/err.txt
#SBATCH -o /homes/mlugli/output/out.txt
#SBATCH --job-name=simclr_head_train
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00

python3 /homes/mlugli/AIBIO_proj/source/train_classifier_norm.py /homes/mlugli/AIBIO_proj/config/train/head_train_norm.yaml
