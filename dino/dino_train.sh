#!/bin/bash
#SBATCH -e /homes/mlugli/output/out3.txt
#SBATCH -o /homes/mlugli/output/err3.txt
#SBATCH --job-name=6c_DINO_3
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"

source activate dinoenv

python3 /homes/mlugli/AIBIO_proj/dino/main_dino.py \
    --multi_centering False \
    --use_original_code False \
    --arch vit_small \
    --saveckp_freq 2\
    --data_path /work/h2020deciderficarra_shared/rxrx1/rxrx1_orig \
    --metadata_path /work/h2020deciderficarra_shared/rxrx1/rxrx1_orig/metadata/meta.csv \
    --output_dir /work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/6c_3 \
    --load_pretrained /work/h2020deciderficarra_shared/rxrx1/checkpoints/OFFICIAL_ViT_pretrained/dino_deitsmall16_pretrain.pth \
    --warmup_teacher_temp_epochs 10 \
    --lr 5e-4 \
    --weight_decay 7e-3 \
    --cell_type HUVEC \
    --num_workers 16 \
    --out_dim 65536 \