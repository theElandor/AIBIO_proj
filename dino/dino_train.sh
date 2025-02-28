#!/bin/bash
#SBATCH -e /homes/mlugli/output/err_3.txt
#SBATCH -o /homes/mlugli/output/out_3.txt
#SBATCH --job-name=backbone_dino
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
python3 /homes/mlugli/AIBIO_proj/dino/main_dino.py --arch vit_small --saveckp_freq 2 --data_path /work/ai4bio2024/rxrx1 --output_dir /work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/cross_batch_reduced
