#!/bin/bash
#SBATCH -e /homes/ocarpentiero/output/err.txt
#SBATCH -o /homes/ocarpentiero/output/out.txt
#SBATCH --job-name=rxrx1_dataset_builder
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=100G
#SBATCH --time=24:00:00

python /homes/ocarpentiero/AIBIO_proj/data_preprocessing/dataset_maker.py