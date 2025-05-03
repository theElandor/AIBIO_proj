#!/bin/bash
#SBATCH -o /homes/mlugli/output/3c_FULL_5_o.txt
#SBATCH -e /homes/mlugli/output/3c_FULL_5_e.txt
#SBATCH --job-name=DINO_3C_5
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source activate dinoenv

python3 /homes/mlugli/AIBIO_proj/dino/main_dino.py \
    --arch vit_small \
    --saveckp_freq 20 \
    --data_path /work/h2020deciderficarra_shared/rxrx1/rxrx1_orig \
    --metadata_path /work/h2020deciderficarra_shared/rxrx1/folds/aug_meta0.csv \
    --output_dir /work/ai4bio2024/rxrx1/check_backup/checkpoints/dino/3c_FULL_5 \
    --load_pretrained /work/ai4bio2024/rxrx1/check_backup/checkpoints/OFFICIAL_ViT_pretrained/dino_deitsmall16_pretrain.pth \
    --epochs 100 \
    --warmup_teacher_temp_epochs 10 \
    --warmup_epochs 10 \
    --lr 5e-4 \
    --weight_decay 4e-3 \
    --cell_type all \
    --num_workers 16 \
    --acc_steps 8 \
    --out_dim 2048 \
    --batch_size_per_gpu 64 \
    --momentum_teacher 0.996 \
    --multi_center_training True \
    --custom_loss True \
    --barlow_loss True \
    --barlow_loss_weight 0.25 \
    --easy_task True \
    --sample_diff_cell_type True \
    --channels 3 \
