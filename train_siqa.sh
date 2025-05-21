#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --model_path="state-spaces/mamba-130m" \
    --tokenizer_path="EleutherAI/gpt-neox-20b" \
    --instruction_datasets="[siqa]" \
    --output_dir="/mnt/server5_hard2/seokil/mamba_output/mamba-FT-out/siqa/130m/1e-4" \
    --random_seed=42 \
    --sequence_max_length=512 \
    --save_steps=5000 \
    --batch_size=4 \
    --cache_dir="huggingface" \
    --num_epochs=12 \
    --weight_decay=0.01 \
    --learning_rate=1e-4 \
    --dropout_rate=0.1 \
    --logging_steps=100