#!/bin/sh
#SBATCH -e /homes/nmorelli/output/err.txt
#SBATCH -o /homes/nmorelli/output/out.txt
#SBATCH --job-name=simclr_train
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --time=24:00:00

python3 /homes/nmorelli/AIBIO_proj/source/train.py /homes/nmorelli/AIBIO_proj/config/train/server_conf.yaml
