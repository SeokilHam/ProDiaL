#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_path="state-spaces/mamba-130m" \
    --tokenizer_path="EleutherAI/gpt-neox-20b" \
    --instruction_datasets="[hellaswag]" \
    --output_dir="outputs" \
    --random_seed=42 \
    --sequence_max_length=512 \
    --save_steps=1 \
    --batch_size=4 \
    --cache_dir="huggingface" \
    --num_epochs=21 \
    --weight_decay=0.01 \
    --learning_rate=1e-4 \
    --dropout_rate=0.1 \
    --logging_steps=100 \
    --config_path="configs/130m" \
    --r_b1=768 \
    --r_b2=1536 \
    --off_diagonal_rank=16 \
