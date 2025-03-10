#!/bin/bash
#SBATCH -e /homes/mlugli/output/out2.txt
#SBATCH -o /homes/mlugli/output/err2.txt
#SBATCH --job-name=warmup_CDCL_full
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G"

source activate dinoenv

python3 /homes/mlugli/AIBIO_proj/dino/main_dino.py \
    --multi_centering True \
    --use_original_code True \
    --arch vit_small \
    --saveckp_freq 2\
    --data_path /work/ai4bio2024/rxrx1 \
    --output_dir /work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/custom_centering_4 \
    --warmup_teacher_temp_epochs 10 \
    --lr 7e-4 \
    --weight_decay 7e-3 \
    --load_pretrained /work/h2020deciderficarra_shared/rxrx1/checkpoints/OFFICIAL_ViT_pretrained/dino_deitsmall16_pretrain.pth
