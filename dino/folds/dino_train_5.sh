#!/bin/bash
#SBATCH -e /homes/mlugli/output/f5o.txt
#SBATCH -o /homes/mlugli/output/f5e.txt
#SBATCH --job-name=DINO_f5
#SBATCH --account=ai4bio2024
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

source activate dinoenv

python3 /homes/mlugli/AIBIO_proj/dino/main_dino.py \
    --multi_centering False \
    --arch vit_small \
    --saveckp_freq 2\
    --data_path /work/h2020deciderficarra_shared/rxrx1/rxrx1_orig \
    --metadata_path /work/h2020deciderficarra_shared/rxrx1/rxrx1_orig/metadatas/meta_5.csv \
    --output_dir /work/h2020deciderficarra_shared/rxrx1/checkpoints/dino/folds/f5 \
    --load_pretrained /work/h2020deciderficarra_shared/rxrx1/checkpoints/OFFICIAL_ViT_pretrained/dino_deitsmall16_pretrain.pth \
    --warmup_teacher_temp_epochs 20 \
    --lr 5e-4 \
    --weight_decay 7e-3 \
    --cell_type HUVEC \
    --num_workers 16 \
    --acc_steps 8 \
    --out_dim 2048 \
    --batch_size_per_gpu 64 \