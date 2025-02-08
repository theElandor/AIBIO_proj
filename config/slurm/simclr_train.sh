#!/bin/sh
#SBATCH -e /homes/nmorelli/output/err.txt
#SBATCH -o /homes/nmorelli/output/out.txt
#SBATCH --job-name=simclr_train
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
SBATCH --nodelist=ailb-login-03

python3 /homes/nmorelli/AIBIO_proj/source/train_backbone.py /homes/nmorelli/AIBIO_proj/config/train/general_conf.yaml
